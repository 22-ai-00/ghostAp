"""Review agent selection helpers for Spec adaptive role review."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Protocol

from src.spec_engine.review_roles import ReviewRoleSpec
from src.spec_engine.review_selection import SpecReviewSelectionItem


class _RandomLike(Protocol):
    def shuffle(self, x: list) -> None: ...
    def choice(self, seq: list): ...


@dataclass(frozen=True)
class ReviewAgentBinding:
    provider: str
    tool_name: str
    display_name: str
    agent_type: str
    model_name: str | None = None
    model_display_name: str | None = None
    selection_key: str = ""
    use_default_model: bool = True

    @property
    def display_label(self) -> str:
        model = self.model_display_name or self.model_name or "默认模型"
        return f"{self.display_name or self.tool_name} / {model}"

    @classmethod
    def from_selection_item(cls, item: SpecReviewSelectionItem) -> "ReviewAgentBinding":
        provider = str(item.provider or "").strip().lower()
        tool_name = str(item.tool_name or "").strip().lower()
        if provider == "ttadk":
            agent_type = f"ttadk_{tool_name}" if tool_name else "ttadk_coco"
        elif provider == "cli":
            agent_type = tool_name or "claude"
        else:
            agent_type = tool_name or "coco"
        return cls(
            provider=provider,
            tool_name=tool_name,
            display_name=str(item.display_name or tool_name or agent_type).strip(),
            agent_type=agent_type,
            model_name=item.model_name,
            model_display_name=item.model_display_name,
            selection_key=item.selection_key,
        )

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "tool_name": self.tool_name,
            "display_name": self.display_name,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "model_display_name": self.model_display_name,
            "selection_key": self.selection_key,
            "use_default_model": self.use_default_model,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ReviewAgentBinding | None":
        if not isinstance(data, dict):
            return None
        agent_type = str(data.get("agent_type") or "").strip().lower()
        tool_name = str(data.get("tool_name") or "").strip().lower()
        if not agent_type and not tool_name:
            return None
        return cls(
            provider=str(data.get("provider") or "").strip().lower(),
            tool_name=tool_name,
            display_name=str(data.get("display_name") or tool_name or agent_type).strip(),
            agent_type=agent_type or tool_name or "coco",
            model_name=str(data.get("model_name") or "").strip() or None,
            model_display_name=str(data.get("model_display_name") or "").strip() or None,
            selection_key=str(data.get("selection_key") or "").strip(),
            use_default_model=data.get("use_default_model", True),
        )


def normalize_review_agents(items: Iterable[object] | None) -> list[ReviewAgentBinding]:
    agents: list[ReviewAgentBinding] = []
    seen: set[str] = set()
    for item in items or []:
        if isinstance(item, ReviewAgentBinding):
            agent = item
        elif isinstance(item, SpecReviewSelectionItem):
            agent = ReviewAgentBinding.from_selection_item(item)
        else:
            agent = ReviewAgentBinding.from_dict(item)
        if not agent:
            continue
        key = agent.selection_key or f"{agent.agent_type}:{agent.model_name or 'default'}"
        if key in seen:
            continue
        seen.add(key)
        agents.append(agent)
    return agents


def assign_review_agents(
    roles: Iterable[ReviewRoleSpec],
    agents: Iterable[ReviewAgentBinding],
    *,
    rng: _RandomLike | None = None,
) -> dict[str, ReviewAgentBinding]:
    """Assign selected review agents to roles.

    The distribution is intentionally random per review cycle, but the first
    pass gives every selected agent a slot when the role count allows it.
    """

    role_list = list(roles or [])
    agent_pool = list(agents or [])
    if not role_list or not agent_pool:
        return {}

    randomizer = rng or random.SystemRandom()
    role_indices = list(range(len(role_list)))
    agent_order = list(agent_pool)
    randomizer.shuffle(role_indices)
    randomizer.shuffle(agent_order)

    assigned: dict[str, ReviewAgentBinding] = {}
    unique_count = min(len(role_indices), len(agent_order))
    for offset in range(unique_count):
        role = role_list[role_indices[offset]]
        assigned[role.role_id] = agent_order[offset]

    for idx in role_indices[unique_count:]:
        role = role_list[idx]
        assigned[role.role_id] = randomizer.choice(agent_pool)

    return assigned
