"""Immutable confirmation-time contract for Workflow execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from src.spec_engine.models import ReviewAgentBinding

from .agent_pool import WorkflowAgentBinding, freeze_agent_pool


def _binding_model(binding: ReviewAgentBinding) -> str | None:
    return None if binding.use_default_model else binding.model_name


def _review_binding(binding: WorkflowAgentBinding) -> ReviewAgentBinding:
    return ReviewAgentBinding(
        provider="workflow",
        tool_name=binding.tool_name,
        display_name=binding.display_name,
        agent_type=binding.tool_name,
        model_name=binding.model_name,
        model_display_name=binding.model_name,
        selection_key=binding.agent_id,
        use_default_model=binding.model_name is None,
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
    legacy_replay: bool = False
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

        if self.legacy_replay:
            self._initialize_legacy(reviewers)
        else:
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

        for reviewer in reviewers:
            reviewer_key = (reviewer.tool_name, _binding_model(reviewer))
            if not any(binding.tool_model_key == reviewer_key for binding in pool):
                raise ValueError("Workflow reviewer binding is outside agent_pool")

        # The legacy map cannot represent two models for one tool. Preserve only
        # unambiguous entries for audit; strict execution always resolves by ID.
        models_by_tool: dict[str, set[str | None]] = {}
        for binding in pool:
            models_by_tool.setdefault(binding.tool_name, set()).add(binding.model_name)
        audit_map = {
            tool: next(iter(models))
            for tool, models in models_by_tool.items()
            if len(models) == 1
        }

        object.__setattr__(self, "agent_pool", pool)
        object.__setattr__(self, "orchestrator_agent_id", orchestrator_agent_id)
        object.__setattr__(self, "orchestrator", _review_binding(orchestrator_binding))
        object.__setattr__(self, "allowed_tools", requested_tools)
        object.__setattr__(self, "tool_model_map", MappingProxyType(audit_map))
        object.__setattr__(self, "_agent_by_id", MappingProxyType(by_id))

    def _initialize_legacy(self, reviewers: tuple[ReviewAgentBinding, ...]) -> None:
        if self.agent_pool:
            raise ValueError("Legacy Workflow replay cannot include agent_pool")
        if self.orchestrator_agent_id:
            raise ValueError("Legacy Workflow replay cannot include orchestrator_agent_id")
        if not self.orchestrator.tool_name:
            raise ValueError("Legacy Workflow orchestrator must name a tool")
        bindings = (self.orchestrator, *reviewers)
        binding_tools = [binding.tool_name for binding in bindings]
        if len(binding_tools) != len(set(binding_tools)):
            raise ValueError("Legacy Workflow replay requires a unique tool per binding")

        normalized_map = {
            str(tool or "").strip(): (str(model).strip() if model else None)
            for tool, model in dict(self.tool_model_map or {}).items()
            if str(tool or "").strip()
        }
        missing_tools = sorted(tool for tool in binding_tools if tool not in normalized_map)
        if missing_tools:
            raise ValueError(
                "Workflow run tool-model map is missing confirmed bindings: "
                + ", ".join(missing_tools)
            )
        mismatched_models = sorted(
            binding.tool_name
            for binding in bindings
            if normalized_map.get(binding.tool_name) != _binding_model(binding)
        )
        if mismatched_models:
            raise ValueError(
                "Workflow tool-model map conflicts with confirmed bindings: "
                + ", ".join(mismatched_models)
            )
        allowed_tools = tuple(
            dict.fromkeys(
                str(tool or "").strip()
                for tool in (self.allowed_tools or tuple(normalized_map))
                if str(tool or "").strip()
            )
        )
        if any(tool not in allowed_tools for tool in binding_tools):
            raise ValueError("Workflow run allowed tools must include every confirmed binding")
        allowed_tool_set = set(allowed_tools)
        frozen_tool_set = set(normalized_map)
        if allowed_tool_set != frozen_tool_set:
            missing = sorted(allowed_tool_set - frozen_tool_set)
            extra = sorted(frozen_tool_set - allowed_tool_set)
            details: list[str] = []
            if missing:
                details.append("missing bindings: " + ", ".join(missing))
            if extra:
                details.append("extra bindings: " + ", ".join(extra))
            raise ValueError(
                "Legacy replay allowed tools must exactly match frozen model "
                "bindings (" + "; ".join(details) + ")"
            )

        object.__setattr__(self, "agent_pool", ())
        object.__setattr__(self, "orchestrator_agent_id", None)
        object.__setattr__(self, "allowed_tools", allowed_tools)
        object.__setattr__(self, "tool_model_map", MappingProxyType(normalized_map))
        object.__setattr__(self, "_agent_by_id", MappingProxyType({}))

    def agent_binding(self, agent_id: str | None) -> WorkflowAgentBinding:
        """Resolve one strict pool member or fail closed before backend dispatch."""

        normalized = str(agent_id or "").strip()
        if self.legacy_replay:
            raise ValueError("Legacy Workflow replay does not resolve agent_id")
        if not normalized:
            raise ValueError("Workflow agent_id is required")
        binding = self._agent_by_id.get(normalized)
        if binding is None:
            raise ValueError(f"Workflow agent_id is outside confirmed agent_pool: {normalized}")
        return binding

    @classmethod
    def from_legacy_replay(cls, **kwargs: Any) -> "WorkflowRunSpec":
        """Explicit boundary for restoring persisted pre-Agent-Pool runs."""

        return cls(**kwargs, legacy_replay=True)

    def to_dict(self) -> dict[str, Any]:
        data = {
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
            "legacy_replay": self.legacy_replay,
        }
        if not self.legacy_replay:
            data["agent_pool"] = [binding.to_dict() for binding in self.agent_pool]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowRunSpec":
        pool_was_provided = "agent_pool" in data
        if pool_was_provided:
            raw_pool_value = data.get("agent_pool")
            if not isinstance(raw_pool_value, (list, tuple)) or not raw_pool_value:
                raise ValueError(
                    "Explicit agent_pool must be a non-empty list of bindings"
                )
        elif data.get("legacy_replay") is False:
            raise ValueError(
                "RunSpec without agent_pool cannot set legacy_replay=false"
            )

        reviewers = tuple(
            reviewer
            for item in (data.get("reviewers") or [])
            if (reviewer := ReviewAgentBinding.from_dict(item)) is not None
        )
        raw_pool = data.get("agent_pool") or []
        if raw_pool:
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
            )

        orchestrator = ReviewAgentBinding.from_dict(data.get("orchestrator"))
        if orchestrator is None:
            raise ValueError("Legacy Workflow replay has no valid orchestrator")
        return cls.from_legacy_replay(
            orchestrator=orchestrator,
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
        )
