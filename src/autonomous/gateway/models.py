"""Immutable, secret-free models for one employee execution attempt."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from ..authorization import EmployeeAuthorizationScope
from ..ingress.targeted_task import TARGETED_TASK_INPUT_KIND

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)\Z"
)
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


class DispatchPermitConsumedError(RuntimeError):
    """A one-shot permit has already been claimed."""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} is required")
    if len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ValueError(f"invalid {name}")
    return value


def _optional_text(value: object, name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"invalid {name}")
    if len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ValueError(f"invalid {name}")
    return value


def _digest(value: object, name: str) -> str:
    text = _required_text(value, name)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _optional_digest(value: object, name: str) -> str:
    text = _optional_text(value, name)
    if text and _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _identifier(value: object, name: str, prefix: str) -> str:
    text = _required_text(value, name)
    if not text.startswith(prefix) or _SAFE_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must use {prefix} identifier space")
    return text


@dataclass(frozen=True, slots=True)
class DispatchBinding:
    """Full replay authority anchored before any external ACP side effect."""

    schema_version: int
    authorization_scope: EmployeeAuthorizationScope
    permit_id: str
    attempt_id: str
    acceptance_id: str
    ingress_aggregate_id: str
    envelope_id: str
    payload_digest: str
    tenant_key: str
    agent_id: str
    employee_version: int
    owner_principal_id: str
    bot_principal_id: str
    app_id: str
    channel_generation: int
    ingress_connection_id: str
    authority_connection_id: str
    requester_principal_id: str
    source_requester_principal_id: str
    task_id: str
    run_id: str
    message_id: str
    thread_root_id: str
    thread_id: str
    chat_id: str
    team_identity: str
    team_chat_id: str
    team_root_identity: str
    tool: str
    model: str
    profile: str
    effort: str
    security_profile: str
    capabilities: tuple[str, ...]
    permissions: tuple[str, ...]
    constraints_digest: str
    system_prompt_token_reserve: int
    render_contract_digest: str
    context_snapshot_hash: str
    context_watermark_digest: str
    effective_input_kind: str
    effective_input_digest: str
    target_bot_open_id_digest: str
    prompt_digest: str
    dispatch_committed_at: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ValueError("unsupported dispatch binding schema")
        if not isinstance(
            self.authorization_scope,
            EmployeeAuthorizationScope,
        ):
            raise TypeError("invalid dispatch authorization scope")
        if type(self.employee_version) is not int or self.employee_version < 0:
            raise ValueError("employee_version must be a non-negative integer")
        for name in ("channel_generation",):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.system_prompt_token_reserve) is not int
            or self.system_prompt_token_reserve < 0
        ):
            raise ValueError("system_prompt_token_reserve must be non-negative")
        for name in (
            "tenant_key",
            "agent_id",
            "owner_principal_id",
            "bot_principal_id",
            "app_id",
            "ingress_connection_id",
            "authority_connection_id",
            "requester_principal_id",
            "source_requester_principal_id",
            "task_id",
            "run_id",
            "message_id",
            "chat_id",
            "team_chat_id",
            "tool",
            "profile",
            "effort",
            "security_profile",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "thread_id", _optional_text(self.thread_id, "thread_id"))
        object.__setattr__(
            self,
            "thread_root_id",
            _optional_text(self.thread_root_id, "thread_root_id"),
        )
        object.__setattr__(self, "model", _optional_text(self.model, "model"))
        if self.security_profile != "employee_v1":
            raise ValueError("dispatch binding requires employee_v1 security profile")
        for name, prefix in (
            ("permit_id", "prm_"),
            ("attempt_id", "att_"),
            ("acceptance_id", "acc_"),
            ("ingress_aggregate_id", "dedup_"),
            ("envelope_id", "ing_"),
            ("agent_id", "agt_"),
            ("bot_principal_id", "bot_"),
            ("app_id", "cli_"),
            ("task_id", "task_"),
            ("run_id", "run_"),
            ("message_id", "om_"),
            ("chat_id", "oc_"),
            ("team_chat_id", "oc_"),
        ):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, prefix),
            )
        if self.thread_root_id:
            object.__setattr__(
                self,
                "thread_root_id",
                _identifier(self.thread_root_id, "thread_root_id", "om_"),
            )
        for name in (
            "payload_digest",
            "team_identity",
            "team_root_identity",
            "render_contract_digest",
            "context_snapshot_hash",
            "context_watermark_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        prompt_digest = _optional_digest(self.prompt_digest, "prompt_digest")
        if self.schema_version == 2 and not prompt_digest:
            raise ValueError("prompt_digest is required for dispatch binding v2")
        if self.schema_version == 1 and prompt_digest:
            raise ValueError("legacy dispatch binding cannot carry prompt_digest")
        object.__setattr__(self, "prompt_digest", prompt_digest)
        effective_input_kind = _optional_text(
            self.effective_input_kind,
            "effective_input_kind",
        )
        effective_input_digest = _optional_digest(
            self.effective_input_digest,
            "effective_input_digest",
        )
        target_bot_open_id_digest = _optional_digest(
            self.target_bot_open_id_digest,
            "target_bot_open_id_digest",
        )
        object.__setattr__(self, "effective_input_kind", effective_input_kind)
        object.__setattr__(self, "effective_input_digest", effective_input_digest)
        object.__setattr__(
            self,
            "target_bot_open_id_digest",
            target_bot_open_id_digest,
        )
        if not (
            bool(effective_input_kind)
            == bool(effective_input_digest)
            == bool(target_bot_open_id_digest)
        ):
            raise ValueError("effective input binding is incomplete")
        if effective_input_kind:
            if (
                effective_input_kind != TARGETED_TASK_INPUT_KIND
                or self.authorization_scope
                is not EmployeeAuthorizationScope.MANAGED_GROUP
                or self.requester_principal_id != self.owner_principal_id
            ):
                raise ValueError("unsupported effective input kind")
        if self.schema_version == 1 and effective_input_kind:
            raise ValueError("legacy dispatch binding cannot carry effective input")
        object.__setattr__(
            self,
            "constraints_digest",
            _optional_digest(self.constraints_digest, "constraints_digest"),
        )
        for name in ("capabilities", "permissions"):
            raw_values = getattr(self, name)
            if isinstance(raw_values, (str, bytes)):
                raise ValueError(f"{name} must be a collection of names")
            values = tuple(sorted(raw_values))
            if any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in values
            ):
                raise ValueError(f"{name} must contain names")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, values)
        if (
            not isinstance(self.dispatch_committed_at, str)
            or _RFC3339_UTC_RE.fullmatch(self.dispatch_committed_at) is None
        ):
            raise ValueError("dispatch_committed_at must be canonical UTC")

    def to_dict(self) -> dict[str, object]:
        if self.schema_version != 2:
            raise ValueError("legacy dispatch binding is replay-only")
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["authorization_scope"] = self.authorization_scope.value
        result["permissions"] = list(self.permissions)
        result["capabilities"] = list(self.capabilities)
        return dict(sorted(result.items()))

    @classmethod
    def from_dict(cls, value: object) -> DispatchBinding:
        if not isinstance(value, dict):
            raise ValueError("dispatch binding must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("dispatch binding must use exact schema")
        if not isinstance(value.get("permissions"), list) or not isinstance(
            value.get("capabilities"),
            list,
        ):
            raise ValueError("dispatch authority collections must be JSON arrays")
        normalized = dict(value)
        if normalized.get("schema_version") != 2:
            raise ValueError("current dispatch binding requires schema version 2")
        try:
            normalized["authorization_scope"] = EmployeeAuthorizationScope(
                normalized["authorization_scope"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid dispatch authorization scope") from exc
        return cls(**normalized)


@dataclass(slots=True)
class _PermitGate:
    lock: threading.Lock = field(default_factory=threading.Lock)
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class DispatchPermit:
    """Ephemeral one-shot execution capability issued after its binding anchors."""

    binding: DispatchBinding
    prompt: str = field(repr=False)
    engine: Any = field(repr=False, compare=False)
    agent: Any = field(repr=False, compare=False)
    timeout_seconds: float
    env: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    _gate: _PermitGate = field(default_factory=_PermitGate, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, DispatchBinding):
            raise TypeError("binding must be DispatchBinding")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt is required")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 0 < float(self.timeout_seconds)
        ):
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(self.env, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in self.env.items()
        ):
            raise ValueError("env must be an explicit string mapping")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))

    def claim(self) -> None:
        with self._gate.lock:
            if self._gate.consumed:
                raise DispatchPermitConsumedError("dispatch permit already consumed")
            self._gate.consumed = True


class GatewayExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    ACTION_REQUIRED = "action_required"


@dataclass(frozen=True, slots=True)
class AgentExecutionSpec:
    """Frozen copy of every Team identity field used by employee execution."""

    agent_id: str
    name: str
    emoji: str
    agent_type: str
    model_name: str
    model_profile: str
    reasoning_effort: str
    system_prompt: str
    role: str
    permissions: tuple[str, ...]
    memory_path: str
    notes_path: str
    workspace_path: str
    owner_group: str
    member_groups: tuple[str, ...]
    created_at: float
    personality_traits: tuple[str, ...]
    capabilities: tuple[str, ...]
    security_profile: str
    wake_policy: str

    @classmethod
    def from_agent(cls, agent: Any) -> AgentExecutionSpec:
        from src.autonomous.workforce.identity import AgentIdentity

        if not isinstance(agent, AgentIdentity):
            raise TypeError("agent must be AgentIdentity")
        return cls(
            agent_id=agent.agent_id,
            name=agent.name,
            emoji=agent.emoji,
            agent_type=agent.agent_type,
            model_name=agent.model_name,
            model_profile=agent.model_profile,
            reasoning_effort=agent.reasoning_effort,
            system_prompt=agent.system_prompt,
            role=agent.role,
            permissions=tuple(agent.permissions),
            memory_path=agent.memory_path,
            notes_path=agent.notes_path,
            workspace_path=agent.workspace_path,
            owner_group=agent.owner_group,
            member_groups=tuple(agent.member_groups),
            created_at=agent.created_at,
            personality_traits=tuple(agent.personality_traits),
            capabilities=tuple(agent.capabilities),
            security_profile=agent.security_profile,
            wake_policy=agent.wake_policy,
        )

    def materialize(self) -> Any:
        from src.autonomous.workforce.identity import AgentIdentity

        return AgentIdentity(
            agent_id=self.agent_id,
            name=self.name,
            emoji=self.emoji,
            agent_type=self.agent_type,
            model_name=self.model_name,
            model_profile=self.model_profile,
            reasoning_effort=self.reasoning_effort,
            system_prompt=self.system_prompt,
            role=self.role,
            permissions=list(self.permissions),
            memory_path=self.memory_path,
            notes_path=self.notes_path,
            workspace_path=self.workspace_path,
            owner_group=self.owner_group,
            member_groups=list(self.member_groups),
            created_at=self.created_at,
            personality_traits=list(self.personality_traits),
            capabilities=list(self.capabilities),
            security_profile=self.security_profile,
            wake_policy=self.wake_policy,
        )


@dataclass(frozen=True, slots=True)
class GatewayExecutionResult:
    status: GatewayExecutionStatus
    output: str = field(default="", repr=False)
    safe_error_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, GatewayExecutionStatus):
            raise TypeError("status must be GatewayExecutionStatus")
        if not isinstance(self.output, str) or not isinstance(self.safe_error_code, str):
            raise TypeError("gateway result text fields must be strings")
        if self.status is GatewayExecutionStatus.COMPLETED and not self.output:
            raise ValueError("completed execution requires output")
        if self.status is not GatewayExecutionStatus.COMPLETED and self.output:
            raise ValueError("non-completed execution cannot carry output")


__all__ = [
    "AgentExecutionSpec",
    "DispatchBinding",
    "DispatchPermit",
    "DispatchPermitConsumedError",
    "GatewayExecutionResult",
    "GatewayExecutionStatus",
]
