"""Default role bootstrapping for slock passive mode.

Creates a predefined set of agents when a slock team is initialized,
ensuring the group is immediately capable of processing user messages
without manual /new-role commands.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from .engine import SlockEngine

from .models import AgentIdentity

logger = logging.getLogger(__name__)

# Supported tool_type values — used for config validation during bootstrap
SUPPORTED_TOOL_TYPES: frozenset[str] = frozenset(
    {"codex", "claude", "coco", "aiden", "gemini"}
)

# Role → default personality traits mapping
_DEFAULT_TRAITS: dict[str, list[str]] = {
    "coder": ["严谨", "注重细节", "高效"],
    "reviewer": ["批判性思维", "注重质量", "全面"],
    "writer": ["清晰表达", "结构化", "用户视角"],
    "tester": ["严谨", "边界思维", "全覆盖"],
    "planner": ["全局观", "结构化思维", "优先级敏感"],
    "architect": ["系统思维", "前瞻性", "权衡利弊"],
}

# Role → default emoji
_DEFAULT_EMOJI: dict[str, str] = {
    "coder": "👨‍💻",
    "reviewer": "🔍",
    "writer": "✍️",
    "tester": "🧪",
    "planner": "📋",
    "architect": "🏗️",
}

# Extended emoji pool per role — used to assign unique emojis when multiple agents share a role
_EMOJI_POOL: dict[str, list[str]] = {
    "coder": ["👨‍💻", "💻", "⌨️", "🔧", "🛠️", "⚡", "🦾", "🧑‍💻"],
    "reviewer": ["🔍", "👁️", "🧐", "📝", "✅", "🎯", "🔎", "📌"],
    "writer": ["✍️", "📖", "🖊️", "📄", "💬", "🗒️", "📚", "🪶"],
    "tester": ["🧪", "🔬", "🐛", "🧫", "🏃", "🎲", "🧩", "⚗️"],
    "planner": ["📋", "🗺️", "📐", "🧭", "📊", "🗓️", "🎯", "📑"],
    "architect": ["🏗️", "🏛️", "🌐", "🧱", "📐", "🔺", "🏰", "🗼"],
    "custom": ["🤖", "🌟", "🎭", "🦊", "🐙", "🌈", "🎪", "🔮"],
}


def pick_unique_emoji(role: str, used_emojis: set[str]) -> str:
    """Pick an emoji for the given role that is not already used.

    Falls back to the default emoji if all pool entries are exhausted.
    """
    pool = _EMOJI_POOL.get(role, _EMOJI_POOL["custom"])
    for e in pool:
        if e not in used_emojis:
            return e
    # All exhausted — fall back to default or first pool entry
    return _DEFAULT_EMOJI.get(role, "🤖")

# Role → default permissions mapping (least privilege principle)
_DEFAULT_PERMISSIONS: dict[str, list[str]] = {
    "coder": ["shell", "file_write", "git"],
    "reviewer": ["file_read"],
    "writer": ["file_read"],
    "tester": ["shell", "file_read"],
    "planner": ["file_read"],
    "architect": ["file_read"],
}


def parse_default_roles(config_str: str) -> list[tuple[str, str]]:
    """Parse slock_default_roles config string into (role_name, tool_type) pairs.

    Format: "coder:codex,reviewer:claude"
    Returns: [("coder", "codex"), ("reviewer", "claude")]
    """
    result = []
    for entry in config_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            role_name, tool_type = entry.split(":", 1)
            result.append((role_name.strip(), tool_type.strip()))
        else:
            # Default to role_name as both role and tool type
            result.append((entry.strip(), entry.strip()))
    return result


def bootstrap_default_roles(
    engine: "SlockEngine",
    channel_id: str,
    config_str: str,
) -> list[AgentIdentity]:
    """Bootstrap default roles for a slock channel.

    Idempotent: skips roles that already exist (by name + channel).
    Returns the list of created (or existing) AgentIdentity objects.
    """
    registry: AgentRegistry = engine.registry
    roles = parse_default_roles(config_str)
    created: list[AgentIdentity] = []

    for role_name, tool_type in roles:
        # Validate tool_type before attempting creation
        if tool_type not in SUPPORTED_TOOL_TYPES:
            logger.warning(
                "bootstrap_default_roles: skipping role '%s' — invalid tool_type '%s' "
                "(supported: %s)",
                role_name, tool_type, ", ".join(sorted(SUPPORTED_TOOL_TYPES)),
            )
            continue

        # Idempotent check: skip if agent with this name already exists in channel
        existing = registry.find_by_name(role_name, channel_id=channel_id)
        if existing is not None:
            logger.debug(
                "bootstrap_default_roles: role '%s' already exists in channel %s, skipping",
                role_name, channel_id,
            )
            created.append(existing)
            continue

        agent_id = f"{tool_type}:default:{uuid.uuid4().hex[:8]}"
        # Pick unique emoji — avoid duplicates with existing agents in registry
        used_emojis = {a.emoji for a in registry.list_agents() if hasattr(a, 'emoji')}
        emoji = pick_unique_emoji(role_name, used_emojis | {a.emoji for a in created if hasattr(a, 'emoji')})
        identity = AgentIdentity(
            agent_id=agent_id,
            name=role_name,
            emoji=emoji,
            agent_type=tool_type,
            role=role_name,
            permissions=_DEFAULT_PERMISSIONS.get(role_name, ["file_read"]),
            owner_group=channel_id,
            personality_traits=_DEFAULT_TRAITS.get(role_name, []),
        )

        try:
            registered = registry.register(identity)
            logger.info(
                "bootstrap_default_roles: created role '%s' (type=%s) in channel %s",
                role_name, tool_type, channel_id,
            )
            created.append(registered)
        except Exception as e:
            logger.warning(
                "bootstrap_default_roles: failed to create role '%s': %s",
                role_name, e,
            )

    # --- Degradation path: no agents created ---
    if not created:
        logger.warning(
            "bootstrap_default_roles: completed with 0 agents for channel %s",
            channel_id,
        )
        # Mark channel as bootstrap_failed
        channel = engine.get_channel()
        if channel is not None:
            channel.bootstrap_failed = True

        # Build and send degradation notification card with actionable hints
        from .card_templates.queue_feedback import build_no_agent_available_card

        card = build_no_agent_available_card(
            team_name=channel_id,
            hint="请检查 SLOCK_DEFAULT_ROLES 配置，或使用 /new-role 手动创建角色。系统将定时重试。",
        )
        try:
            engine.send_card(card)
        except Exception as exc:
            logger.debug(
                "bootstrap_default_roles: failed to send degradation card: %s", exc
            )

        # Schedule retry: mark pending tasks for retry in 5 minutes
        # The dispatch loop will check bootstrap_failed and retry automatically
        if engine.task_queue is not None:
            import time
            for task in engine.task_queue.snapshot():
                if task.status == "pending" and task.bootstrap_pending:
                    task.retry_count = getattr(task, "retry_count", 0) + 1
                    task.next_retry_at = time.time() + 300  # 5 minutes

    return created
