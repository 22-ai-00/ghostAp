"""Backend-neutral bootstrap contract for one logical employee session."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from src.autonomous.workforce.identity import AgentIdentity


@dataclass(frozen=True, slots=True)
class EmployeeSessionKey:
    tenant_key: str
    agent_id: str
    project_root: str
    backend: str
    model: str
    profile: str
    effort: str
    identity_version: int
    instruction_digest: str


@dataclass(frozen=True, slots=True)
class EmployeeSessionBootstrap:
    session_key: EmployeeSessionKey
    project_root: str
    workspace_root: str
    codex_home: str
    instruction_text: str
    instruction_digest: str

    @classmethod
    def from_agent(
        cls,
        *,
        tenant_key: str,
        agent: AgentIdentity,
        project_root: str,
        identity_version: int,
    ) -> "EmployeeSessionBootstrap":
        project = os.path.realpath(project_root)
        workspace = os.path.realpath(agent.workspace_path)
        if not tenant_key or not project or not workspace:
            raise ValueError("employee session coordinates are required")
        instruction_path = Path(workspace) / "AGENTS.md"
        instruction_bytes = instruction_path.read_bytes()
        if not instruction_bytes or len(instruction_bytes) > 8192:
            raise ValueError("employee bootstrap instruction is invalid")
        if (
            isinstance(identity_version, bool)
            or not isinstance(identity_version, int)
            or identity_version < 0
        ):
            raise ValueError("employee identity version is invalid")
        instruction = instruction_bytes.decode("utf-8")
        instruction_digest = hashlib.sha256(instruction_bytes).hexdigest()
        employee_root = Path(workspace).parent
        codex_home = os.path.realpath(employee_root / "runtime" / "codex-home")
        return cls(
            session_key=EmployeeSessionKey(
                tenant_key=tenant_key,
                agent_id=agent.agent_id,
                project_root=project,
                backend=agent.agent_type,
                model=agent.model_name,
                profile=agent.model_profile,
                effort=agent.reasoning_effort,
                identity_version=identity_version,
                instruction_digest=instruction_digest,
            ),
            project_root=project,
            workspace_root=workspace,
            codex_home=codex_home,
            instruction_text=instruction,
            instruction_digest=instruction_digest,
        )

    def wrap_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("employee prompt is required")
        return (
            "## GHOSTAP_EMPLOYEE_BOOTSTRAP\n"
            f"identity={self.session_key.agent_id}\n"
            f"instruction_digest={self.instruction_digest}\n"
            f"workspace={self.workspace_root}\n\n"
            f"{self.instruction_text}\n\n"
            "## ASSIGNMENT\n"
            f"{prompt}"
        )


__all__ = ["EmployeeSessionBootstrap", "EmployeeSessionKey"]
