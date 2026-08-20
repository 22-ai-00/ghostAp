"""Execution identity and persistent employee storage coordinates."""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field

AGENT_ROLE_COLORS: dict[str, str] = {
    "coder": "blue",
    "writer": "green",
    "reviewer": "orange",
    "tester": "purple",
    "planner": "red",
    "architect": "indigo",
    "custom": "grey",
}


def default_employee_storage_base() -> str:
    """Return the existing employee data root without moving user data."""
    # Keep the legacy `~/.ghostap/slock` path string for compatibility with
    # historical autonomous employee data migrations.
    return os.path.expanduser("~/.ghostap/slock")


@dataclass
class AgentIdentity:
    """Backend-neutral employee identity projected from the Journal."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    emoji: str = "🤖"
    agent_type: str = "coco"
    model_name: str = ""
    model_profile: str = "standard"
    reasoning_effort: str = "default"
    system_prompt: str = ""
    role: str = "custom"
    permissions: list[str] = field(default_factory=lambda: ["shell", "file_write", "git"])
    memory_path: str = ""
    notes_path: str = ""
    workspace_path: str = ""
    owner_group: str = ""
    member_groups: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    personality_traits: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    wake_policy: str = ""

    def __post_init__(self) -> None:
        self.agent_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", self.agent_id)
        if ".." in self.agent_id or self.agent_id.startswith("."):
            self.agent_id = self.agent_id.lstrip(".").replace("..", "_")
        if self.owner_group and self.owner_group not in self.member_groups:
            self.member_groups.append(self.owner_group)

    @property
    def display_name(self) -> str:
        return f"{self.emoji} {self.name}" if self.name else f"{self.emoji} Agent"

    @property
    def card_color(self) -> str:
        return AGENT_ROLE_COLORS.get(self.role, "grey")

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "emoji": self.emoji,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "model_profile": self.model_profile,
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": self.system_prompt,
            "role": self.role,
            "permissions": list(self.permissions),
            "memory_path": self.memory_path,
            "notes_path": self.notes_path,
            "workspace_path": self.workspace_path,
            "owner_group": self.owner_group,
            "member_groups": list(self.member_groups),
            "created_at": self.created_at,
            "personality_traits": list(self.personality_traits),
            "capabilities": list(self.capabilities),
            "wake_policy": self.wake_policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentIdentity":
        return cls(**{
            key: value
            for key, value in data.items()
            if key in cls.__dataclass_fields__
        })


__all__ = ["AGENT_ROLE_COLORS", "AgentIdentity", "default_employee_storage_base"]
