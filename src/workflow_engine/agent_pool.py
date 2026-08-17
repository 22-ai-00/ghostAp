"""Immutable identities for user-confirmed Workflow Agent bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class WorkflowAgentBinding:
    """One stable Agent identity bound to an exact backend tool and model."""

    agent_id: str
    tool_name: str
    model_name: str | None
    display_name: str
    profile: str | None = None
    effort: str | None = None

    def __post_init__(self) -> None:
        agent_id = str(self.agent_id or "").strip()
        tool_name = str(self.tool_name or "").strip().lower()
        model_name = str(self.model_name or "").strip() or None
        display_name = str(self.display_name or "").strip()
        profile = str(self.profile or "").strip() or None
        effort = str(self.effort or "").strip() or None
        if not _AGENT_ID_RE.fullmatch(agent_id):
            raise ValueError(
                "Workflow Agent agent_id must be 1-128 safe identifier characters"
            )
        if not tool_name:
            raise ValueError("Workflow Agent tool_name must not be empty")
        if not display_name:
            raise ValueError("Workflow Agent display_name must not be empty")
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "effort", effort)

    @property
    def tool_model_key(self) -> tuple[str, str | None]:
        return self.tool_name, self.model_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "profile": self.profile,
            "effort": self.effort,
        }

    @classmethod
    def from_dict(cls, data: object) -> "WorkflowAgentBinding | None":
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                agent_id=str(data.get("agent_id") or data.get("agentId") or ""),
                tool_name=str(data.get("tool_name") or data.get("toolName") or ""),
                model_name=(
                    str(data.get("model_name") or data.get("modelName") or "").strip()
                    or None
                ),
                display_name=str(
                    data.get("display_name")
                    or data.get("displayName")
                    or data.get("agent_id")
                    or data.get("agentId")
                    or ""
                ),
                profile=str(data.get("profile") or "").strip() or None,
                effort=str(data.get("effort") or "").strip() or None,
            )
        except ValueError:
            return None


def freeze_agent_pool(
    bindings: Iterable[WorkflowAgentBinding | dict[str, Any]],
    *,
    require_nonempty: bool = True,
) -> tuple[WorkflowAgentBinding, ...]:
    """Normalize an ordered pool and reject ambiguous identities or bindings."""

    pool: list[WorkflowAgentBinding] = []
    for raw in bindings:
        binding = raw if isinstance(raw, WorkflowAgentBinding) else WorkflowAgentBinding.from_dict(raw)
        if binding is None:
            raise ValueError("Workflow agent_pool contains an invalid binding")
        pool.append(binding)
    if require_nonempty and not pool:
        raise ValueError("Workflow agent_pool must not be empty")

    seen_ids: set[str] = set()
    seen_tool_models: set[tuple[str, str | None]] = set()
    for binding in pool:
        if binding.agent_id in seen_ids:
            raise ValueError(f"Workflow agent_pool has duplicate agent_id: {binding.agent_id}")
        if binding.tool_model_key in seen_tool_models:
            tool, model = binding.tool_model_key
            raise ValueError(
                "Workflow agent_pool has duplicate tool/model binding: "
                f"{tool}/{model or 'default'}"
            )
        seen_ids.add(binding.agent_id)
        seen_tool_models.add(binding.tool_model_key)
    return tuple(pool)


def select_auto_orchestrator(
    bindings: Iterable[WorkflowAgentBinding | dict[str, Any]],
    *,
    recommendations: Sequence[dict[str, Any]] = (),
    preferred_tools: Sequence[str] = (
        "traex",
        "claude",
        "codex",
        "grok",
        "dsh",
        "aiden",
        "gemini",
        "coco",
    ),
) -> WorkflowAgentBinding:
    """Resolve Auto independently of pool insertion order."""
    pool = freeze_agent_pool(bindings)
    exact_rank: dict[tuple[str, str | None], int] = {}
    recommended_tool_rank: dict[str, int] = {}
    for index, raw in enumerate(recommendations):
        tool = str(raw.get("tool_name") or raw.get("toolName") or "").strip().lower()
        if not tool:
            continue
        model = str(raw.get("model_name") or raw.get("modelName") or "").strip() or None
        exact_rank.setdefault((tool, model), index)
        recommended_tool_rank.setdefault(tool, index)
    preferred_rank = {
        str(tool).strip().lower(): index for index, tool in enumerate(preferred_tools)
    }

    def agent_id_rank(agent_id: str) -> tuple[int, str]:
        match = re.fullmatch(r"A-?(\d+)", agent_id, re.IGNORECASE)
        return (int(match.group(1)), "") if match else (10**9, agent_id)

    sentinel = 10**9
    return min(
        pool,
        key=lambda binding: (
            exact_rank.get(binding.tool_model_key, sentinel),
            recommended_tool_rank.get(binding.tool_name, sentinel),
            preferred_rank.get(binding.tool_name, sentinel),
            0 if binding.model_name is None else 1,
            binding.model_name or "",
            agent_id_rank(binding.agent_id),
        ),
    )
