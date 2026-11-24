"""Immutable confirmation-time contract for Workflow execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from src.spec_engine.models import ReviewAgentBinding

from .agent_pool import WorkflowAgentBinding, freeze_agent_pool, resolve_effective_model_name


def _binding_model(binding: ReviewAgentBinding) -> str | None:
    return None if binding.use_default_model else binding.model_name


def _is_default_effective_sentinel(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text == "default" or text.startswith("default/"):
        return True
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return False
        return (
            isinstance(parsed, list)
            and len(parsed) >= 2
            and parsed[0] == "default"
            and parsed[1] in {"", "default"}
        )
    return False


def _review_binding(
    binding: WorkflowAgentBinding,
    effective_model: str | None = None,
) -> ReviewAgentBinding:
    resolved_effective = (
        effective_model if effective_model is not None else binding.effective_model_name()
    )
    resolved_model = resolved_effective or binding.model_name
    return ReviewAgentBinding(
        provider="workflow",
        tool_name=binding.tool_name,
        display_name=binding.display_name,
        agent_type=binding.tool_name,
        model_name=resolved_model,
        model_display_name=binding.model_name,
        selection_key=binding.agent_id,
        use_default_model=(resolved_effective is None and binding.model_name is None),
    )


@dataclass(frozen=True, slots=True)
class WorkflowRunSpec:
    """Frozen Agent binding, identity, and resource contract for one run."""

    orchestrator: ReviewAgentBinding
    reviewers: tuple[ReviewAgentBinding, ...]
    tool_model_map: Mapping[str, str | None]
    task: str
    chat_id: str
    topic_id: str | None
    budget: int
    deadline: float | None
    auto_reviewer: bool
    initiator_user_id: str | None = None
    allowed_tools: tuple[str, ...] = ()
    enforce_tool_allowlist: bool = True
    agent_pool: tuple[WorkflowAgentBinding, ...] = ()
    orchestrator_agent_id: str | None = None
    binding_specs: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    _agent_by_id: Mapping[str, WorkflowAgentBinding] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        task = str(self.task or "").strip()
        chat_id = str(self.chat_id or "").strip()
        if not task:
            raise ValueError("Workflow run task must not be empty")
        if not chat_id:
            raise ValueError("Workflow run chat_id must not be empty")
        if self.budget <= 0:
            raise ValueError("Workflow run budget must be positive")
        if self.deadline is not None and self.deadline <= 0:
            raise ValueError("Workflow run deadline must be positive when set")

        reviewers = tuple(self.reviewers or ())
        if self.auto_reviewer and reviewers:
            raise ValueError("Auto reviewer mode cannot include explicit reviewers")
        if not self.auto_reviewer and not reviewers:
            raise ValueError("Explicit review mode requires at least one reviewer")

        self._initialize_strict_pool(reviewers)

        object.__setattr__(self, "task", task)
        object.__setattr__(self, "chat_id", chat_id)
        object.__setattr__(self, "topic_id", str(self.topic_id).strip() if self.topic_id else None)
        object.__setattr__(self, "reviewers", reviewers)

    def _initialize_strict_pool(self, reviewers: tuple[ReviewAgentBinding, ...]) -> None:
        pool = freeze_agent_pool(self.agent_pool)
        by_id = {binding.agent_id: binding for binding in pool}
        orchestrator_agent_id = str(self.orchestrator_agent_id or "").strip() or pool[0].agent_id
        if orchestrator_agent_id not in by_id:
            raise ValueError(
                "Workflow orchestrator_agent_id is not in agent_pool: "
                f"{orchestrator_agent_id}"
            )
        orchestrator_binding = by_id[orchestrator_agent_id]

        pool_tools = tuple(dict.fromkeys(binding.tool_name for binding in pool))
        requested_tools = tuple(
            dict.fromkeys(
                str(tool or "").strip()
                for tool in (self.allowed_tools or pool_tools)
                if str(tool or "").strip()
            )
        )
        if set(requested_tools) != set(pool_tools):
            raise ValueError("Workflow allowed_tools must exactly match strict agent_pool tools")

        effective_by_id: dict[str, str | None] = {}
        raw_binding_specs = (
            self.binding_specs if isinstance(self.binding_specs, Mapping) else {}
        )
        for binding in pool:
            computed = binding.effective_model_name()
            resolved_effective = computed
            if binding.model_name is None and _is_default_effective_sentinel(computed):
                raw_spec = raw_binding_specs.get(binding.agent_id)
                saved = (
                    raw_spec.get("effective_model")
                    if isinstance(raw_spec, Mapping)
                    else None
                )
                if (
                    isinstance(saved, str)
                    and saved.strip()
                    and not _is_default_effective_sentinel(saved.strip())
                    and resolve_effective_model_name(
                        binding.tool_name,
                        saved.strip(),
                        profile=binding.profile,
                        effort=binding.effort,
                    )
                    == saved.strip()
                ):
                    resolved_effective = saved.strip()
            effective_by_id[binding.agent_id] = resolved_effective

        for reviewer in reviewers:
            reviewer_model = _binding_model(reviewer)
            reviewer_id = str(getattr(reviewer, "selection_key", "") or "").strip()
            require_agent_id = bool(reviewer_id and ":" not in reviewer_id)
            if not any(
                (not require_agent_id or binding.agent_id == reviewer_id)
                and binding.tool_name == reviewer.tool_name
                and reviewer_model in {binding.model_name, effective_by_id[binding.agent_id]}
                for binding in pool
            ):
                raise ValueError("Workflow reviewer binding is outside agent_pool")

        # The legacy map cannot represent two configs (model/profile/effort) for
        # one tool. Preserve only unambiguous entries for audit; strict execution
        # always resolves by agent_id via binding_specs / _agent_by_id.
        configs_by_tool: dict[
            str, dict[tuple[str | None, str | None, str | None], str | None]
        ] = {}
        for binding in pool:
            config_key = (binding.model_name, binding.profile, binding.effort)
            configs_by_tool.setdefault(binding.tool_name, {})[config_key] = (
                effective_by_id[binding.agent_id]
            )
        audit_map = {
            tool: next(iter(configs.values()))
            for tool, configs in configs_by_tool.items()
            if len(configs) == 1
        }
        frozen_binding_specs = MappingProxyType(
            {
                binding.agent_id: MappingProxyType(
                    {
                        **binding.to_dict(),
                        "roles": tuple(binding.roles),
                        "effective_model": effective_by_id[binding.agent_id],
                    }
                )
                for binding in pool
            }
        )

        object.__setattr__(self, "agent_pool", pool)
        object.__setattr__(self, "orchestrator_agent_id", orchestrator_agent_id)
        object.__setattr__(
            self,
            "orchestrator",
            _review_binding(orchestrator_binding, effective_by_id[orchestrator_agent_id]),
        )
        object.__setattr__(self, "allowed_tools", requested_tools)
        object.__setattr__(self, "tool_model_map", MappingProxyType(audit_map))
        object.__setattr__(self, "binding_specs", frozen_binding_specs)
        object.__setattr__(self, "_agent_by_id", MappingProxyType(by_id))

    def agent_binding(self, agent_id: str | None) -> WorkflowAgentBinding:
        """Resolve one strict pool member or fail closed before backend dispatch."""

        normalized = str(agent_id or "").strip()
        if not normalized:
            raise ValueError("Workflow agent_id is required")
        binding = self._agent_by_id.get(normalized)
        if binding is None:
            raise ValueError(f"Workflow agent_id is outside confirmed agent_pool: {normalized}")
        return binding

    def to_dict(self) -> dict[str, Any]:
        return {
            "orchestrator": self.orchestrator.to_dict(),
            "reviewers": [reviewer.to_dict() for reviewer in self.reviewers],
            "tool_model_map": dict(self.tool_model_map),
            "task": self.task,
            "chat_id": self.chat_id,
            "topic_id": self.topic_id,
            "budget": self.budget,
            "deadline": self.deadline,
            "auto_reviewer": self.auto_reviewer,
            "initiator_user_id": self.initiator_user_id,
            "allowed_tools": list(self.allowed_tools),
            "enforce_tool_allowlist": self.enforce_tool_allowlist,
            "orchestrator_agent_id": self.orchestrator_agent_id,
            "agent_pool": [binding.to_dict() for binding in self.agent_pool],
            "binding_specs": {
                aid: {
                    **dict(spec),
                    "roles": list(spec.get("roles") or ()),
                }
                for aid, spec in self.binding_specs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowRunSpec":
        raw_pool = data.get("agent_pool")
        if not isinstance(raw_pool, (list, tuple)) or not raw_pool:
            raise ValueError("Explicit agent_pool must be a non-empty list of bindings")

        reviewers = tuple(
            reviewer
            for item in (data.get("reviewers") or [])
            if (reviewer := ReviewAgentBinding.from_dict(item)) is not None
        )
        pool = freeze_agent_pool(raw_pool)
        orchestrator_id = str(data.get("orchestrator_agent_id") or "").strip() or None
        selected_id = orchestrator_id or pool[0].agent_id
        selected = next(
            (binding for binding in pool if binding.agent_id == selected_id),
            pool[0],
        )
        return cls(
            orchestrator=_review_binding(selected),
            reviewers=reviewers,
            tool_model_map=dict(data.get("tool_model_map") or {}),
            task=str(data.get("task") or ""),
            chat_id=str(data.get("chat_id") or ""),
            topic_id=str(data.get("topic_id") or "").strip() or None,
            budget=int(data.get("budget") or 0),
            deadline=(float(data["deadline"]) if data.get("deadline") is not None else None),
            auto_reviewer=bool(data.get("auto_reviewer")),
            initiator_user_id=str(data.get("initiator_user_id") or "").strip() or None,
            allowed_tools=tuple(data.get("allowed_tools") or ()),
            enforce_tool_allowlist=bool(data.get("enforce_tool_allowlist", True)),
            agent_pool=pool,
            orchestrator_agent_id=orchestrator_id,
            binding_specs=dict(data.get("binding_specs") or {}),
        )
