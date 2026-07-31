"""Immutable confirmation-time contract for Workflow execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from src.spec_engine.review_agents import ReviewAgentBinding


@dataclass(frozen=True, slots=True)
class WorkflowRunSpec:
    """Frozen binding and resource contract captured by the handler once.

    Mutable execution progress belongs to ``WorkflowProject``. This value is
    passed unchanged from confirmation to the engine, bridge, ordinary agent
    calls, and committed reviewer calls.
    """

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

    def __post_init__(self) -> None:
        task = str(self.task or "").strip()
        chat_id = str(self.chat_id or "").strip()
        if not task:
            raise ValueError("Workflow run task must not be empty")
        if not chat_id:
            raise ValueError("Workflow run chat_id must not be empty")
        if not self.orchestrator.tool_name:
            raise ValueError("Workflow run orchestrator must name a tool")
        if self.budget <= 0:
            raise ValueError("Workflow run budget must be positive")
        if self.deadline is not None and self.deadline <= 0:
            raise ValueError("Workflow run deadline must be positive when set")

        reviewers = tuple(self.reviewers or ())
        if self.auto_reviewer and reviewers:
            raise ValueError("Auto reviewer mode cannot include explicit reviewers")
        if not self.auto_reviewer and not reviewers:
            raise ValueError("Explicit review mode requires at least one reviewer")

        normalized_map = {
            str(tool or "").strip(): (str(model).strip() if model else None)
            for tool, model in dict(self.tool_model_map or {}).items()
            if str(tool or "").strip()
        }
        bound_tools = {
            self.orchestrator.tool_name,
            *(reviewer.tool_name for reviewer in reviewers),
        }
        missing_tools = sorted(tool for tool in bound_tools if tool not in normalized_map)
        if missing_tools:
            raise ValueError(
                "Workflow run tool-model map is missing confirmed bindings: "
                + ", ".join(missing_tools)
            )

        confirmed_models: dict[str, str | None] = {}
        for binding in (self.orchestrator, *reviewers):
            if not binding.use_default_model and not binding.model_name:
                raise ValueError(
                    f"Workflow confirmed model is empty for tool {binding.tool_name}"
                )
            confirmed_models.setdefault(
                binding.tool_name,
                None if binding.use_default_model else binding.model_name,
            )
        mismatched_models = sorted(
            tool
            for tool, model in confirmed_models.items()
            if normalized_map.get(tool) != model
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
        if any(tool not in allowed_tools for tool in bound_tools):
            raise ValueError("Workflow run allowed tools must include every confirmed binding")

        object.__setattr__(self, "task", task)
        object.__setattr__(self, "chat_id", chat_id)
        object.__setattr__(self, "topic_id", str(self.topic_id).strip() if self.topic_id else None)
        object.__setattr__(self, "reviewers", reviewers)
        object.__setattr__(self, "tool_model_map", MappingProxyType(normalized_map))
        object.__setattr__(self, "allowed_tools", allowed_tools)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe audit representation."""
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
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowRunSpec":
        """Restore a persisted run contract without weakening validation."""
        orchestrator = ReviewAgentBinding.from_dict(data.get("orchestrator"))
        if orchestrator is None:
            raise ValueError("Workflow run spec has no valid orchestrator")
        reviewers = tuple(
            reviewer
            for item in (data.get("reviewers") or [])
            if (reviewer := ReviewAgentBinding.from_dict(item)) is not None
        )
        return cls(
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
