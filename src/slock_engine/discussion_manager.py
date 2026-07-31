"""Inter-agent discussion protocol for the Slock multi-agent collaboration engine.

This module implements the discussion manager that orchestrates structured
conversations between agents — handling trigger detection, round execution,
convergence checking, token budget control, and conclusion summarization.

Thread-safety note: Discussion execution runs within the engine's bounded
executor. The DiscussionThread dataclass is mutable and should only be
mutated by one coroutine at a time (ensured by the engine's task scheduling).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Optional

from .models import (
    AgentIdentity,
    DiscussionConfig,
    DiscussionMessage,
    DiscussionStatus,
    DiscussionThread,
)
from .protocols import DiscussionEngineProtocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum content length for discussion messages (64KB default)
MAX_DISCUSSION_CONTENT_LENGTH = 64 * 1024  # 64KB

# Regex pattern for @mention detection
AT_MENTION_PATTERN: re.Pattern[str] = re.compile(r"@([\w-]+)")

# Signals that indicate discussion convergence (agents agreeing)
CONVERGENCE_SIGNALS: set[str] = {
    "AGREE",
    "LGTM",
    "同意",
    "认可",
    "没问题",
    "looks good",
    "sounds good",
    "no further suggestions",
}

# Markers that indicate uncertainty in agent output (triggers discussion)
UNCERTAINTY_MARKERS: set[str] = {
    "不确定",
    "需要确认",
    "需要讨论",
    "needs review",
    "需要审查",
    "not sure",
    "i'm not sure",
    "uncertain",
    "maybe",
    "可能",
    "也许",
}

# Antonym pairs for knowledge conflict detection (rule-based)
# Each tuple contains mutually exclusive concepts
_CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("mysql", "postgresql"),
    ("postgres", "mysql"),
    ("allow", "deny"),
    ("permit", "block"),
    ("enable", "disable"),
    ("true", "false"),
    ("yes", "no"),
    ("always", "never"),
    ("must", "must not"),
    ("required", "optional"),
    ("include", "exclude"),
    ("add", "remove"),
    ("create", "delete"),
    ("public", "private"),
    ("http", "https"),
    ("sync", "async"),
    ("synchronous", "asynchronous"),
    ("increase", "decrease"),
    ("upgrade", "downgrade"),
    ("accept", "reject"),
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DiscussionCancelledError(Exception):
    """Raised when a discussion is cancelled via cooperative cancellation event."""
    pass


# ---------------------------------------------------------------------------
# DiscussionManager
# ---------------------------------------------------------------------------


class DiscussionManager:
    """Orchestrates inter-agent discussions within the Slock engine.

    Responsibilities:
    - Detect when a discussion should be triggered based on rules, @mentions,
      or uncertainty markers in agent output.
    - Manage the discussion lifecycle: start, execute rounds, check convergence,
      summarize conclusions.
    - Enforce token budget constraints to prevent runaway conversations.
    """

    def __init__(
        self,
        *,
        engine: Optional[DiscussionEngineProtocol] = None,
        memory_manager: Any = None,
        config: Optional[DiscussionConfig] = None,
        on_unavailable_notify: Optional[Any] = None,
        on_budget_warning: Optional[Any] = None,
    ) -> None:
        """Initialize the discussion manager.

        Args:
            engine: Reference to SlockEngine (must satisfy DiscussionEngineProtocol).
            memory_manager: Reference to MemoryManager for context retrieval.
            config: Default discussion configuration. Uses DiscussionConfig defaults
                    if not provided.
            on_unavailable_notify: Optional callback(agent_id, reason) called when an
                    agent is unavailable during discussion (for user notification).
            on_budget_warning: Optional callback(thread) called when token usage
                    reaches 80% of the budget.
        """
        # Type annotation provides static checking; runtime accepts duck-typed objects
        self._engine: Optional[DiscussionEngineProtocol] = engine  # type: ignore[assignment]
        self._memory_manager = memory_manager
        self._config = config or DiscussionConfig()
        self._on_unavailable_notify = on_unavailable_notify
        self._on_budget_warning = on_budget_warning

        # Task 26: Cooldown and depth limit tracking
        self._last_discussion_time: dict[str, float] = {}
        self._discussion_depth: dict[str, int] = {}
        self._cooldown_seconds = 60.0
        self._force_cooldown_seconds = 10.0  # Shorter cooldown even for forced triggers
        self._max_depth = 3
        if self._engine is not None:
            try:
                self._max_depth = getattr(
                    self._engine.settings, 'slock_max_discussion_depth', 3
                )
            except (AttributeError, TypeError):
                pass

        # Card update debounce: prevent exceeding Feishu API rate limits (1 update/sec)
        self._last_card_update_time: float = 0.0
        self._pending_card_update: Optional[Any] = None  # thread awaiting deferred update

        # Task 27: Discussion-task binding
        self._task_bindings: dict[str, str] = {}  # thread_id -> task_id

        logger.info(
            "DiscussionManager initialized (max_rounds=%d, token_budget=%d)",
            self._config.max_rounds,
            self._config.token_budget,
        )

    # ------------------------------------------------------------------
    # Dynamic configuration properties (hot-reload from settings)
    # ------------------------------------------------------------------

    @property
    def uncertainty_markers(self) -> tuple[str, ...]:
        """Read uncertainty markers from settings (hot-reloadable)."""
        from ..config import get_settings
        try:
            return tuple(get_settings().slock_uncertainty_markers)
        except AttributeError:
            return tuple(UNCERTAINTY_MARKERS)

    @property
    def convergence_signals(self) -> tuple[str, ...]:
        """Read convergence signals from settings (hot-reloadable)."""
        from ..config import get_settings
        try:
            return tuple(get_settings().slock_convergence_signals)
        except AttributeError:
            return tuple(CONVERGENCE_SIGNALS)

    # ------------------------------------------------------------------
    # Adaptive Discussion Governance (Self-Assessment Gate)
    # ------------------------------------------------------------------

    def _assess_collaboration_need(self, agent_id: str, content: str) -> float:
        """Heuristic assessment: 0.0-1.0 score for discussion need.

        < 0.3: confident, proceed alone (skip discussion trigger)
        0.3-0.6: mild uncertainty, sidebar notification sufficient
        >= 0.6: genuine uncertainty, formal discussion warranted
        """
        score = 0.0

        # 1. Uncertainty marker density in content
        window = content[-500:] if len(content) > 500 else content
        uncertainty_words = [
            "不确定", "可能", "也许", "需要确认", "不太清楚", "有疑问",
            "unsure", "maybe", "unclear", "not sure", "needs review",
            "uncertain", "需要讨论", "需要检查",
        ]
        marker_count = sum(1 for m in uncertainty_words if m in window)
        score += min(marker_count * 0.15, 0.35)

        # 2. Question density
        question_marks = window.count("?") + window.count("？")
        if question_marks >= 3:
            score += 0.15
        elif question_marks >= 1:
            score += 0.05

        # 3. Content length as complexity proxy
        if len(content) > 3000:
            score += 0.1

        # 4. Contradictory signals (agent expressing two opposing views)
        contradiction_pairs = [("但是", "然而"), ("不过", "虽然"), ("however", "but")]
        contradiction_count = sum(
            1 for pair in contradiction_pairs
            if any(w in window for w in pair)
        )
        if contradiction_count >= 2:
            score += 0.15

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Trigger Detection
    # ------------------------------------------------------------------

    def _get_agent_task_blockers_context(self, agent_id: str, *, channel_id: str = "") -> str:
        """Get task blocker/predecessor context for an agent.

        Returns a string describing any task dependencies (predecessors,
        chain context) that should be considered during discussion.
        Returns empty string if no blockers found.
        """
        if self._engine is None:
            return ""

        try:
            task_mgr = getattr(self._engine, "_task_mgr", None)
            if task_mgr is None:
                return ""

            # Access tasks from task manager
            tasks = getattr(task_mgr, "_tasks", [])
            if not tasks:
                return ""

            # Find the agent's current task (claimed or recently completed)
            agent_task = None
            for task in tasks:
                if getattr(task, "claimed_by", None) == agent_id:
                    agent_task = task
                    break

            if agent_task is None:
                return ""

            context_parts: list[str] = []

            # Check for predecessor info
            predecessor = getattr(agent_task, "predecessor_agent_name", "")
            if predecessor:
                context_parts.append(f"predecessor={predecessor}")

            # Check for chain context in task content
            content = getattr(agent_task, "content", "")
            if content.startswith("[chain:"):
                # Extract chain info like "[chain:coder->reviewer]"
                end_idx = content.find("]")
                if end_idx > 0:
                    chain_info = content[7:end_idx]  # len("[chain:") = 7
                    context_parts.append(f"chain={chain_info}")

            # Check for other pending tasks that might be blockers
            # (tasks in TODO/IN_PROGRESS that this task might depend on)
            pending_count = sum(
                1 for t in tasks
                if getattr(t, "status", None) and getattr(t, "status").value in ("todo", "in_progress")
                and getattr(t, "task_id", "") != getattr(agent_task, "task_id", "")
            )
            if pending_count > 0:
                context_parts.append(f"pending_tasks={pending_count}")

            if context_parts:
                return "[" + ", ".join(context_parts) + "]"

        except Exception as exc:
            logger.debug("Error getting task blockers context: %s", str(exc))

        return ""

    def _build_thread_topic(self, base_topic: str, agent_id: str, *, channel_id: str = "") -> str:
        """Build a discussion thread topic with optional blockers context.

        Appends task blocker context to the base topic if available.
        Truncates to 100 chars max.
        """
        blockers_ctx = self._get_agent_task_blockers_context(agent_id, channel_id=channel_id)
        if blockers_ctx:
            full_topic = f"{base_topic} {blockers_ctx}"
        else:
            full_topic = base_topic
        return full_topic[:100]

    def should_trigger_discussion(
        self,
        agent: AgentIdentity,
        result_content: str,
        config: Optional[DiscussionConfig] = None,
        *,
        force: bool = False,
        channel_id: str = "",
    ) -> Optional[DiscussionThread]:
        """Determine whether an agent's output should trigger a discussion.

        Three trigger strategies are evaluated in order:
        1. Rule trigger — role-based routing (e.g. "coder->reviewer").
        2. @mention trigger — explicit @AgentName in content.
        3. Uncertainty trigger — presence of uncertainty markers.

        Args:
            agent: The agent that produced the result.
            result_content: The text output from the agent's execution.
            config: Optional override config; defaults to instance config.
            force: If True, bypass cooldown and force uncertainty trigger.
            channel_id: The channel where this discussion would take place.

        Returns:
            A new DiscussionThread if triggered, or None.
        """
        cfg = config or self._config

        # Cooldown check (skipped when force=True)
        if not force and self._is_on_cooldown(agent.agent_id):
            logger.debug("Discussion suppressed: agent %s is on cooldown", agent.agent_id)
            return None

        if force:
            # Force mode still respects a shorter anti-spam cooldown
            last_time = self._last_discussion_time.get(agent.agent_id, 0)
            if (time.time() - last_time) < self._force_cooldown_seconds:
                logger.debug("Force-discussion suppressed: agent %s within force cooldown", agent.agent_id)
                return None
            # Force mode: go directly to uncertainty trigger
            thread = self._check_uncertainty_trigger(agent, result_content, cfg, channel_id=channel_id)
            if thread is not None:
                logger.info("Discussion force-triggered (uncertainty) for agent %s", agent.agent_id)
                return thread
            # Fallback: create a generic discussion thread for the agent
            thread = self._create_forced_discussion_thread(agent, result_content, cfg, channel_id=channel_id)
            return thread

        # Self-assessment gate: skip discussion trigger for confident outputs.
        # Rule-based and @mention triggers represent explicit collaboration signals
        # and bypass this gate. The gate prevents spurious uncertainty-only triggers.
        governance_score = self._assess_collaboration_need(agent.agent_id, result_content)

        # --- Strategy 1: Rule-based trigger (explicit, bypasses gate) ---
        thread = self._check_rule_trigger(agent, cfg, channel_id=channel_id)
        if thread is not None:
            logger.info(
                "Discussion triggered by rule for agent %s (role=%s)",
                agent.agent_id,
                agent.role,
            )
            return thread

        # --- Strategy 2: @mention trigger (explicit, bypasses gate) ---
        thread = self._check_mention_trigger(agent, result_content, cfg, channel_id=channel_id)
        if thread is not None:
            logger.info(
                "Discussion triggered by @mention in output of agent %s",
                agent.agent_id,
            )
            return thread

        # Gate check: skip uncertainty trigger for confident outputs
        if governance_score < 0.3:
            logger.debug(
                "Governance gate: skipping discussion trigger (score=%.2f < 0.3)",
                governance_score,
            )
            return None

        # --- Strategy 3: Uncertainty trigger (gated by self-assessment) ---
        thread = self._check_uncertainty_trigger(agent, result_content, cfg, channel_id=channel_id)
        if thread is not None:
            logger.info(
                "Discussion triggered by uncertainty markers in agent %s output",
                agent.agent_id,
            )
            return thread

        return None

    def _check_rule_trigger(
        self, agent: AgentIdentity, config: DiscussionConfig, *, channel_id: str = ""
    ) -> Optional[DiscussionThread]:
        """Check if the agent's role matches any trigger rule."""
        # Fallback for backward compatibility: use agent.owner_group or "default"
        effective_channel = channel_id or getattr(agent, "owner_group", "") or "default"
        for rule in config.trigger_rules:
            parts = rule.split("->")
            if len(parts) != 2:
                logger.warning("Invalid trigger rule format: %s", rule)
                continue
            source_role, target_role = parts[0].strip(), parts[1].strip()
            if agent.role == source_role:
                target_agent_id = self._find_agent_by_role(target_role)
                if target_agent_id is not None:
                    base_topic = f"rule:{rule}"
                    thread = DiscussionThread(
                        thread_id=str(uuid.uuid4()),
                        channel_id=effective_channel,
                        participants=[agent.agent_id, target_agent_id],
                        config=config,
                        trigger_reason=base_topic,
                        topic=self._build_thread_topic(base_topic, agent.agent_id, channel_id=effective_channel),
                    )
                    return thread
                else:
                    logger.debug(
                        "Rule '%s' matched but no agent with role '%s' found",
                        rule,
                        target_role,
                    )
        return None

    def _check_mention_trigger(
        self, agent: AgentIdentity, content: str, config: DiscussionConfig, *, channel_id: str = ""
    ) -> Optional[DiscussionThread]:
        """Check if content contains @AgentName mentions.

        Supports both agent name mentions (@AgentName) and role mentions
        (@Architect, @Reviewer, etc.) with case-insensitive matching.
        """
        # Fallback for backward compatibility: use agent.owner_group or "default"
        effective_channel = channel_id or getattr(agent, "owner_group", "") or "default"
        matches = AT_MENTION_PATTERN.findall(content)
        if not matches:
            return None

        for mentioned_name in matches:
            # First, try to find by exact agent name
            target_agent_id = self._find_agent_by_name(mentioned_name, channel_id=effective_channel)
            if target_agent_id is not None:
                base_topic = f"mention:@{mentioned_name}"
                thread = DiscussionThread(
                    thread_id=str(uuid.uuid4()),
                    channel_id=effective_channel,
                    participants=[agent.agent_id, target_agent_id],
                    config=config,
                    trigger_reason=base_topic,
                    topic=self._build_thread_topic(base_topic, agent.agent_id, channel_id=effective_channel),
                )
                return thread

            # Fallback: try to find by role (case-insensitive)
            target_agent_id = self._find_agent_by_role(mentioned_name.lower(), channel_id=effective_channel)
            if target_agent_id is not None:
                base_topic = f"mention:@role:{mentioned_name}"
                thread = DiscussionThread(
                    thread_id=str(uuid.uuid4()),
                    channel_id=effective_channel,
                    participants=[agent.agent_id, target_agent_id],
                    config=config,
                    trigger_reason=base_topic,
                    topic=self._build_thread_topic(base_topic, agent.agent_id, channel_id=effective_channel),
                )
                return thread

        return None

    def _check_uncertainty_trigger(
        self, agent: AgentIdentity, content: str, config: DiscussionConfig, *, channel_id: str = ""
    ) -> Optional[DiscussionThread]:
        """Check if content contains uncertainty markers and find a discussion partner.

        Only the last 500 characters of *content* are scanned to reduce false
        positives from lengthy agent output.  The marker list is read from
        settings (``slock_uncertainty_markers``) and compiled into a single
        regex pattern for efficient matching.
        """
        from ..config import get_settings

        markers = list(get_settings().slock_uncertainty_markers)
        if not markers:
            return None
        if "unsure" not in markers:
            markers.append("unsure")

        # Build a single regex from the configured markers (case-insensitive)
        escaped = [re.escape(m) for m in markers]
        pattern = "|".join(escaped)

        # Only inspect the tail of the output to reduce false positives
        window = content[-500:]

        match = re.search(pattern, window, re.IGNORECASE)
        if match is None:
            return None

        matched_marker = match.group(0)

        # Use dynamic partner selection instead of hardcoded roles
        # Fallback for backward compatibility: use agent.owner_group or "default"
        effective_channel = channel_id or getattr(agent, "owner_group", "") or "default"
        target_agent_id = self._find_best_discussion_partner(
            agent, content, channel_id=effective_channel
        )
        if target_agent_id is None:
            logger.debug(
                "Uncertainty detected but no suitable partner found"
            )
            return None
        thread = DiscussionThread(
            thread_id=str(uuid.uuid4()),
            channel_id=effective_channel,
            participants=[agent.agent_id, target_agent_id],
            config=config,
            trigger_reason=f"uncertainty:{matched_marker}",
            topic=self._build_thread_topic(f"uncertainty:{matched_marker}", agent.agent_id, channel_id=effective_channel),
        )
        return thread

    def _create_forced_discussion_thread(
        self, agent: AgentIdentity, content: str, config: DiscussionConfig, *, channel_id: str = ""
    ) -> Optional[DiscussionThread]:
        """Create a discussion thread when force-triggered but no uncertainty marker matched."""
        # Fallback for backward compatibility: use agent.owner_group or "default"
        effective_channel = channel_id or getattr(agent, "owner_group", "") or "default"
        target_agent_id = self._find_best_discussion_partner(
            agent, content, channel_id=effective_channel
        )
        if target_agent_id is None:
            logger.debug("Force-triggered discussion but no suitable partner found")
            return None
        base_topic = "forced:uncertainty_detected"
        return DiscussionThread(
            thread_id=str(uuid.uuid4()),
            channel_id=effective_channel,
            participants=[agent.agent_id, target_agent_id],
            config=config,
            trigger_reason=base_topic,
            topic=self._build_thread_topic(base_topic, agent.agent_id, channel_id=effective_channel),
        )

    # ------------------------------------------------------------------
    # Token Estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count with CJK-aware weighted formula.

        Uses a per-character weighted estimate instead of a binary threshold:
        - CJK characters: ~1.5 tokens/char
        - Non-CJK characters: ~0.25 tokens/char (≈ len//4 for pure Latin)

        This eliminates the discontinuity at the 0.3 CJK-ratio boundary and
        provides smooth, accurate estimates for mixed-language content.
        """
        if not content:
            return 0
        cjk_count = sum(1 for ch in content if self._is_cjk_char(ch))
        non_cjk_count = len(content) - cjk_count
        return int(cjk_count * 1.5 + non_cjk_count * 0.25)

    @staticmethod
    def _is_cjk_char(ch: str) -> bool:
        """Check if a character is a CJK ideograph."""
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
            or (0x3400 <= cp <= 0x4DBF)    # CJK Unified Ideographs Extension A
            or (0xF900 <= cp <= 0xFAFF)    # CJK Compatibility Ideographs
            or (0x20000 <= cp <= 0x2A6DF)  # CJK Unified Ideographs Extension B
            or (0x2A700 <= cp <= 0x2B73F)  # CJK Unified Ideographs Extension C
            or (0x2B740 <= cp <= 0x2B81F)  # CJK Unified Ideographs Extension D
            or (0x2B820 <= cp <= 0x2CEAF)  # CJK Unified Ideographs Extension E
            or (0x3000 <= cp <= 0x303F)    # CJK Symbols and Punctuation
            or (0xFF00 <= cp <= 0xFFEF)    # Halfwidth and Fullwidth Forms
        )

    # ------------------------------------------------------------------
    # Cooldown & Depth Limit (Task 26)
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, agent_id: str) -> bool:
        """Check if agent is on discussion cooldown."""
        last_time = self._last_discussion_time.get(agent_id, 0)
        effective = getattr(self, '_cooldown_seconds_effective', self._cooldown_seconds)
        return (time.time() - last_time) < effective

    def _record_discussion_participation(self, agent_id: str, status: Optional["DiscussionStatus"] = None) -> None:
        """Record that an agent participated in a discussion.

        Applies adaptive cooldown based on how the discussion ended:
        - CONVERGED: 15s (quick resolution, allow fast re-engagement)
        - TIMEOUT / MAX_ROUNDS_REACHED: 60s (moderate backoff)
        - BUDGET_EXHAUSTED: 120s (longest, prevent resource drain)
        - Other/None: uses default _cooldown_seconds (60s)
        """
        if status == DiscussionStatus.CONVERGED:
            cooldown = 15.0
        elif status in (DiscussionStatus.TIMEOUT, DiscussionStatus.MAX_ROUNDS_REACHED):
            cooldown = 60.0
        elif status == DiscussionStatus.BUDGET_EXHAUSTED:
            cooldown = 120.0
        else:
            cooldown = self._cooldown_seconds
        self._cooldown_seconds_effective = cooldown
        self._last_discussion_time[agent_id] = time.time()

    def _check_depth_limit(self, parent_thread_id: Optional[str] = None) -> bool:
        """Check if discussion depth limit would be exceeded.

        Returns True if within limits, False if exceeded.
        """
        if parent_thread_id is None:
            return True
        depth = self._discussion_depth.get(parent_thread_id, 0)
        return depth < self._max_depth

    def _increment_depth(self, thread_id: str, parent_thread_id: Optional[str] = None) -> None:
        """Increment the depth counter for a new sub-discussion."""
        parent_depth = self._discussion_depth.get(parent_thread_id, 0) if parent_thread_id else 0
        self._discussion_depth[thread_id] = parent_depth + 1

    # ------------------------------------------------------------------
    # Discussion-Task Binding (Task 27)
    # ------------------------------------------------------------------

    def bind_to_task(self, thread_id: str, task_id: str) -> None:
        """Bind a discussion thread to a specific task."""
        self._task_bindings[thread_id] = task_id
        logger.debug("Discussion %s bound to task %s", thread_id, task_id)

    def get_bound_task(self, thread_id: str) -> Optional[str]:
        """Get the task_id bound to a discussion thread."""
        return self._task_bindings.get(thread_id)

    def unbind_task(self, thread_id: str) -> None:
        """Remove task binding when discussion completes."""
        self._task_bindings.pop(thread_id, None)

    # ------------------------------------------------------------------
    # Discussion Lifecycle
    # ------------------------------------------------------------------

    def _validate_discussion_content(self, content: str) -> str:
        """Validate and sanitize discussion content.

        - Truncates content exceeding MAX_DISCUSSION_CONTENT_LENGTH
        - Removes null bytes and control characters that could cause issues
        - Returns the sanitized content

        Args:
            content: The raw discussion content to validate.

        Returns:
            Sanitized content string.
        """
        if not content:
            return ""

        # Remove null bytes and control characters (except newline and tab)
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)

        # Truncate if too long
        if len(sanitized) > MAX_DISCUSSION_CONTENT_LENGTH:
            logger.warning(
                "Discussion content exceeds max length (%d > %d), truncating",
                len(sanitized),
                MAX_DISCUSSION_CONTENT_LENGTH,
            )
            sanitized = sanitized[:MAX_DISCUSSION_CONTENT_LENGTH] + "... [TRUNCATED]"

        return sanitized

    def _find_bound_task(self, thread_id: str) -> Any:
        task_id = self.get_bound_task(thread_id)
        if not task_id or self._engine is None:
            return None
        tasks = getattr(self._engine, "_tasks", None)
        if tasks is None:
            task_mgr = getattr(self._engine, "_task_mgr", None)
            tasks = getattr(task_mgr, "_tasks", None) if task_mgr is not None else None
        if not tasks:
            return None
        for task in tasks:
            if getattr(task, "task_id", "") == task_id:
                return task
        return None

    def _truncate_task_context_value(self, value: Any, limit: int = 300) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[:limit] + "... [TRUNCATED]"

    def _inject_bound_task_context(self, thread: DiscussionThread, content: str) -> str:
        task = self._find_bound_task(thread.thread_id)
        if task is None:
            return content

        status = getattr(task, "status", "")
        status_text = getattr(status, "value", status)
        context = (
            "=== 关联任务上下文 ===\n"
            f"任务ID: {getattr(task, 'task_id', '')}\n"
            f"任务描述: {self._truncate_task_context_value(getattr(task, 'content', ''))}\n"
            f"当前状态: {status_text}\n"
            f"推理快照: {self._truncate_task_context_value(getattr(task, 'reasoning_snapshot', ''))}\n"
            f"认领者: {getattr(task, 'claimed_by', '') or '(无)'}\n"
            "=== 当前讨论内容 ===\n"
        )
        return context + content

    def start_discussion(
        self, thread: DiscussionThread, initial_content: str
    ) -> DiscussionThread:
        """Start a discussion by adding the initial message and activating the thread.

        Args:
            thread: The discussion thread (from should_trigger_discussion).
            initial_content: The content that triggered the discussion.

        Returns:
            The activated thread with the initial message appended.
        """
        if not thread.participants:
            logger.error("Cannot start discussion with no participants")
            return thread

        # Validate, enrich, and sanitize initial content
        initial_content = self._validate_discussion_content(initial_content)
        initial_content = self._inject_bound_task_context(thread, initial_content)
        initial_content = self._validate_discussion_content(initial_content)

        initial_message = DiscussionMessage(
            message_id=str(uuid.uuid4()),
            sender_agent_id=thread.participants[0],
            receiver_agent_id=(
                thread.participants[1] if len(thread.participants) > 1 else ""
            ),
            content=initial_content,
            round_num=0,
            timestamp=time.time(),
            token_count=self._estimate_tokens(initial_content),
        )

        thread.messages.append(initial_message)
        thread.total_tokens_used += initial_message.token_count
        thread.status = DiscussionStatus.ACTIVE

        logger.info(
            "Discussion %s started with %d participants",
            thread.thread_id[:8],
            len(thread.participants),
        )
        return thread

    def execute_round(self, thread: DiscussionThread) -> DiscussionThread:
        """Execute one round of discussion — determine next speaker and get response.

        Alternates between participants. Builds context from the last 3 messages
        and asks the responding agent to provide their perspective.

        Checks cancellation_event before ACP call for cooperative cancellation.

        Args:
            thread: The active discussion thread.

        Returns:
            The thread with the new response message appended.

        Raises:
            DiscussionCancelledError: If cancellation_event is set (e.g., by watchdog).
        """
        if not thread.is_active:
            logger.warning(
                "Cannot execute round on inactive thread %s (status=%s)",
                thread.thread_id[:8],
                thread.status.value,
            )
            return thread

        # Check for cooperative cancellation before executing
        cancellation_event = thread.cancellation_event
        if cancellation_event and cancellation_event.is_set():
            raise DiscussionCancelledError(
                f"Discussion {thread.thread_id[:8]} cancelled before round execution"
            )

        # Determine next respondent (alternate between participants)
        current_round = thread.current_round + 1
        num_participants = len(thread.participants)
        if num_participants < 2:
            logger.error("Discussion requires at least 2 participants")
            return thread

        # Simple alternation: round 0 = participant[0], round 1 = participant[1], etc.
        respondent_idx = current_round % num_participants
        respondent_id = thread.participants[respondent_idx]

        # Determine the other agent (sender of request)
        sender_idx = (current_round - 1) % num_participants
        sender_id = thread.participants[sender_idx]

        # Build discussion prompt with recent context
        prompt = self._build_round_prompt(thread, respondent_id)

        # Check cancellation again right before the ACP call (most expensive operation)
        if cancellation_event and cancellation_event.is_set():
            raise DiscussionCancelledError(
                f"Discussion {thread.thread_id[:8]} cancelled before ACP call"
            )

        # Execute agent turn
        response, token_count = self._execute_agent_turn(respondent_id, prompt, thread=thread)

        if response is None:
            logger.warning("Discussion aborted: agent turn returned None")
            thread.status = DiscussionStatus.TIMEOUT
            return thread

        # Validate and sanitize response
        response = self._validate_discussion_content(response)

        # Per-round token hard cap enforcement
        max_per_round = getattr(thread.config, 'max_tokens_per_round', 8000)
        if token_count > max_per_round:
            logger.warning(
                "Discussion %s round %d: agent %s exceeded per-round token cap "
                "(%d > %d), forcing termination",
                thread.thread_id[:8],
                current_round,
                respondent_id[:8],
                token_count,
                max_per_round,
            )
            # Still record the message but force-terminate after
            message = DiscussionMessage(
                message_id=str(uuid.uuid4()),
                sender_agent_id=respondent_id,
                receiver_agent_id=sender_id,
                content=response,
                round_num=current_round,
                timestamp=time.time(),
                token_count=token_count,
            )
            thread.messages.append(message)
            thread.total_tokens_used += token_count
            thread.status = DiscussionStatus.BUDGET_EXHAUSTED
            return thread

        # Create response message
        message = DiscussionMessage(
            message_id=str(uuid.uuid4()),
            sender_agent_id=respondent_id,
            receiver_agent_id=sender_id,
            content=response,
            round_num=current_round,
            timestamp=time.time(),
            token_count=token_count,
        )

        thread.messages.append(message)
        thread.total_tokens_used += token_count

        logger.debug(
            "Discussion %s round %d: agent %s responded (%d tokens)",
            thread.thread_id[:8],
            current_round,
            respondent_id[:8],
            token_count,
        )
        return thread

    def _resolve_agent_display(self, agent_id: str) -> str:
        """Resolve agent_id to 'emoji name' display string.

        Looks up the agent identity via the engine's public API.
        Falls back to agent_id[:8] if lookup fails.

        Args:
            agent_id: The agent ID to resolve.

        Returns:
            Display string in 'emoji name' format, or agent_id[:8] as fallback.
        """
        if not agent_id:
            return "unknown"
        if self._engine is not None:
            try:
                identity = self._engine.get_agent(agent_id)
                if identity is not None:
                    emoji = getattr(identity, "emoji", "") or ""
                    name = getattr(identity, "name", "") or ""
                    if emoji or name:
                        return f"{emoji} {name}".strip()
            except Exception:
                pass
        return agent_id[:8]

    def _build_round_prompt(
        self, thread: DiscussionThread, respondent_id: str
    ) -> str:
        """Build the discussion prompt for the next respondent.

        Includes the agent's role context, last 3 messages, instructions,
        and any pending user hints (injected and cleared before this round).
        """
        # Collect last 3 messages for context
        recent_messages = thread.messages[-3:]
        context_lines: list[str] = []

        context_lines.append("=== Discussion Context ===")
        context_lines.append(f"Thread: {thread.thread_id[:8]}")
        context_lines.append(f"Trigger: {thread.trigger_reason}")
        context_lines.append(f"Round: {thread.current_round + 1}/{thread.config.max_rounds}")
        context_lines.append("")

        # Inject pending user hints if any
        pending_hints = thread.consume_hints()
        if pending_hints:
            context_lines.append("--- User Hints (人工干预) ---")
            for i, hint in enumerate(pending_hints, 1):
                context_lines.append(f"[Hint {i}] {hint}")
            context_lines.append("")
            context_lines.append("Please carefully consider the above user hints in your response.")
            context_lines.append("")

        context_lines.append("--- Recent Messages ---")

        for msg in recent_messages:
            sender_label = self._resolve_agent_display(msg.sender_agent_id)
            context_lines.append(f"[{sender_label}] (round {msg.round_num}):")
            context_lines.append(msg.content)
            context_lines.append("")

        context_lines.append("--- Your Turn ---")
        context_lines.append(
            "Review the above discussion and provide your perspective. "
            "If you agree with the conclusion, state 'AGREE'. "
            "If you have improvements, provide them concisely."
        )

        return "\n".join(context_lines)

    # ------------------------------------------------------------------
    # Convergence & Budget
    # ------------------------------------------------------------------

    def check_convergence(self, thread: DiscussionThread) -> bool:
        """Check whether the discussion has converged.

        Convergence is detected when:
        1. The last message contains explicit convergence signals (AGREE, LGTM, etc.)
        2. The last two consecutive messages (regardless of sender) are highly similar
           (exceeding convergence_threshold) — indicates both agents expressing similar views.
        3. The last two messages from the same agent are highly similar
           (indicates the agent is repeating itself — no new information).

        Args:
            thread: The discussion thread to check.

        Returns:
            True if the discussion has converged, False otherwise.
        """
        if len(thread.messages) < 2:
            return False

        last_msg = thread.messages[-1]
        last_content_upper = last_msg.content.upper()

        # Check explicit convergence signals in the last message
        for signal in self.convergence_signals:
            if signal.upper() in last_content_upper:
                logger.debug(
                    "Convergence detected: signal '%s' in last message", signal
                )
                return True

        # Check if last two consecutive messages (rounds) are highly similar
        # This detects when both agents are expressing similar views
        # (e.g., agent A proposes, agent B agrees with similar content)
        prev_msg = thread.messages[-2]
        consecutive_similarity = self._calculate_text_similarity(
            prev_msg.content, last_msg.content
        )
        if consecutive_similarity > thread.config.convergence_threshold:
            logger.debug(
                "Convergence detected: consecutive rounds similarity %.2f > threshold %.2f",
                consecutive_similarity,
                thread.config.convergence_threshold,
            )
            return True

        # Check if last two messages from the same agent are highly similar
        # (indicates the agent is repeating itself — no new information)
        same_sender_messages = [
            m for m in thread.messages if m.sender_agent_id == last_msg.sender_agent_id
        ]
        if len(same_sender_messages) >= 2:
            prev_same_msg = same_sender_messages[-2]
            # Skip if we already checked this pair above
            if prev_same_msg is prev_msg:
                # Already checked in consecutive check above
                pass
            else:
                similarity = self._calculate_text_similarity(
                    prev_same_msg.content, last_msg.content
                )
                if similarity > thread.config.convergence_threshold:
                    logger.debug(
                        "Convergence detected: same-sender similarity %.2f > threshold %.2f",
                        similarity,
                        thread.config.convergence_threshold,
                    )
                    return True

        return False

    def check_budget(self, thread: DiscussionThread) -> bool:
        """Check whether the token budget is still available.

        Args:
            thread: The discussion thread.

        Returns:
            True if budget remains, False if exhausted.
        """
        has_budget = thread.total_tokens_used < thread.config.token_budget
        if not has_budget:
            logger.info(
                "Token budget exhausted for thread %s (%d/%d)",
                thread.thread_id[:8],
                thread.total_tokens_used,
                thread.config.token_budget,
            )

        # Budget warning at 80%
        warning_threshold = thread.config.token_budget * 0.8
        if (thread.total_tokens_used >= warning_threshold
                and not getattr(thread, '_budget_warning_sent', False)):
            thread._budget_warning_sent = True
            if self._on_budget_warning:
                try:
                    self._on_budget_warning(thread)
                except Exception as e:
                    logger.debug("Budget warning callback failed: %s", str(e))

        return has_budget

    def check_budget_with_breaker(
        self,
        thread: "DiscussionThread",
        settings,
        on_card_send=None,
    ) -> bool:
        """Unified budget circuit breaker — checks both token budget and round limit.

        Combines engine-level budget enforcement with discussion-level budget check.
        When a limit is hit, forcibly terminates the discussion by setting status and
        cancellation_event, then sends a breaker notification card via callback.

        Args:
            thread: The discussion thread to check.
            settings: Application settings (needs slock_discussion_token_budget,
                      slock_max_discussion_rounds).
            on_card_send: Optional callback(card_dict) to send breaker card.

        Returns:
            True if the discussion should continue, False if breaker tripped.
        """
        from .models import DiscussionStatus

        token_limit = settings.slock_discussion_token_budget
        round_limit = settings.slock_max_discussion_rounds

        # Token budget breaker
        if thread.total_tokens_used >= token_limit:
            thread.status = DiscussionStatus.BUDGET_EXHAUSTED
            if thread.cancellation_event is not None:
                thread.cancellation_event.set()
            if on_card_send:
                try:
                    on_card_send({
                        "schema": "2.0",
                        "config": {"wide_screen_mode": True},
                        "header": {
                            "title": {"tag": "plain_text", "content": "\u26a0\ufe0f 讨论熔断：Token 预算耗尽"},
                            "template": "orange",
                        },
                        "body": {"elements": [{"tag": "markdown", "content": (
                            f"讨论已强制终止。\n\n"
                            f"- **已消耗**: {thread.total_tokens_used:,} tokens\n"
                            f"- **预算上限**: {token_limit:,} tokens\n"
                            f"- **轮次**: {thread.current_round}\n"
                            f"- **原因**: Token 消耗达到预算上限"
                        )}]},
                    })
                except Exception:
                    pass
            logger.warning(
                "Discussion %s breaker tripped: token budget exhausted (%d/%d)",
                thread.thread_id[:8], thread.total_tokens_used, token_limit,
            )
            return False

        # Round limit breaker
        if thread.current_round >= round_limit:
            thread.status = DiscussionStatus.TIMEOUT
            if thread.cancellation_event is not None:
                thread.cancellation_event.set()
            if on_card_send:
                try:
                    on_card_send({
                        "schema": "2.0",
                        "config": {"wide_screen_mode": True},
                        "header": {
                            "title": {"tag": "plain_text", "content": "\u26a0\ufe0f 讨论熔断：轮数上限"},
                            "template": "orange",
                        },
                        "body": {"elements": [{"tag": "markdown", "content": (
                            f"讨论已强制终止。\n\n"
                            f"- **已执行轮数**: {thread.current_round}\n"
                            f"- **轮数上限**: {round_limit}\n"
                            f"- **Token 消耗**: {thread.total_tokens_used:,}\n"
                            f"- **原因**: 轮数达到上限"
                        )}]},
                    })
                except Exception:
                    pass
            logger.warning(
                "Discussion %s breaker tripped: max rounds reached (%d/%d)",
                thread.thread_id[:8], thread.current_round, round_limit,
            )
            return False

        # Also run the per-thread config budget check (80% warning etc.)
        self.check_budget(thread)
        return True

    # ------------------------------------------------------------------
    # Conclusion & Summary
    # ------------------------------------------------------------------

    def summarize_conclusion(self, thread: DiscussionThread) -> str:
        """Generate a conclusion summary for the completed discussion.

        Attempts to use LLM summarization; falls back to a structured
        summary of the last message and participant list.

        Args:
            thread: The completed discussion thread.

        Returns:
            The conclusion summary text.
        """
        if not thread.messages:
            conclusion = "No messages in discussion."
            thread.conclusion = conclusion
            thread.completed_at = time.time()
            return conclusion

        # Build summary prompt
        all_content = "\n".join(
            f"[Round {m.round_num}] {m.sender_agent_id[:8]}: {m.content}"
            for m in thread.messages
        )
        prompt = (
            "Summarize the following inter-agent discussion. "
            "Provide the key points and final conclusion in 2-3 sentences:\n\n"
            f"{all_content}"
        )

        # Attempt LLM summarization (placeholder)
        conclusion = self._call_llm_for_summary(prompt)

        thread.conclusion = conclusion
        thread.completed_at = time.time()

        logger.info(
            "Discussion %s concluded (status=%s, rounds=%d, tokens=%d)",
            thread.thread_id[:8],
            thread.status.value,
            thread.current_round,
            thread.total_tokens_used,
        )
        return conclusion

    # ------------------------------------------------------------------
    # Full Discussion Loop
    # ------------------------------------------------------------------

    def run_discussion(
        self, thread: DiscussionThread, initial_content: str,
        *, on_round_complete=None, on_round_start=None,
    ) -> DiscussionThread:
        """Run the full discussion loop until convergence, timeout, or budget exhaustion.

        Steps:
        1. Start discussion with initial content.
        2. Loop up to max_rounds:
           a. Check budget — break if exhausted.
           b. Call on_round_start callback if provided (thinking indicator).
           c. Execute a round.
           d. Call on_round_complete callback if provided (with debounce).
           e. Check convergence — break if converged.
        3. If loop ends without convergence, mark as TIMEOUT.
        4. Summarize conclusion.

        Args:
            thread: The discussion thread to execute.
            initial_content: The content that initiated the discussion.
            on_round_complete: Optional callback(thread) invoked after each round.
            on_round_start: Optional callback(thread, respondent_id) invoked before each round.

        Returns:
            The completed discussion thread.
        """
        thread = self.start_discussion(thread, initial_content)

        max_rounds = thread.config.max_rounds

        # Resolve debounce interval from config or engine settings
        debounce_interval: float = 1.5
        if self._engine:
            debounce_interval = getattr(self._engine.settings, 'slock_card_debounce_interval', 1.5)

        def _debounced_round_complete(t):
            """Wrap on_round_complete with debounce — skip if too soon after last update."""
            if on_round_complete is None:
                return
            now = time.monotonic()
            if now - self._last_card_update_time < debounce_interval:
                # Too soon — defer this update, will be flushed at end or next eligible round
                self._pending_card_update = t
                return
            self._last_card_update_time = now
            self._pending_card_update = None
            on_round_complete(t)

        try:
            for round_num in range(max_rounds):
                # Check budget before executing round
                if not self.check_budget(thread):
                    thread.status = DiscussionStatus.BUDGET_EXHAUSTED
                    self.summarize_conclusion(thread)
                    logger.info(
                        "Discussion %s ended: budget exhausted at round %d",
                        thread.thread_id[:8],
                        round_num,
                    )
                    break

                # Compute next respondent for on_round_start callback
                current_round = thread.current_round + 1
                num_participants = len(thread.participants)
                respondent_idx = current_round % num_participants
                respondent_id = thread.participants[respondent_idx] if num_participants >= 2 else ""

                # Notify caller that a round is about to start (thinking indicator)
                if on_round_start and respondent_id:
                    try:
                        on_round_start(thread, respondent_id)
                    except Exception as start_exc:
                        logger.debug("on_round_start callback error: %s", str(start_exc))

                # Execute round
                thread = self.execute_round(thread)

                # Abort if agent turn returned None (thread marked as TIMEOUT)
                if not thread.is_active:
                    logger.warning("Discussion aborted: thread no longer active after round")
                    break

                # Notify caller of round completion (with debounce)
                try:
                    _debounced_round_complete(thread)
                except Exception as cb_exc:
                    logger.debug("on_round_complete callback error: %s", str(cb_exc))

                # Check convergence after round
                if self.check_convergence(thread):
                    thread.status = DiscussionStatus.CONVERGED
                    self.summarize_conclusion(thread)
                    logger.info(
                        "Discussion %s converged at round %d",
                        thread.thread_id[:8],
                        round_num + 1,
                    )
                    break
            else:
                # Loop completed without convergence — max rounds exhausted
                thread.status = DiscussionStatus.MAX_ROUNDS_REACHED
                self.summarize_conclusion(thread)
                logger.info(
                    "Discussion %s reached max rounds (%d) without convergence",
                    thread.thread_id[:8],
                    max_rounds,
                )
                # Attempt forced convergence via Final Arbiter
                self._run_final_arbiter(thread)

                # If arbiter failed to produce a conclusion, escalate to Council
                if not thread.conclusion or len(thread.conclusion.strip()) < 20:
                    self._escalate_to_council(thread)

            # Persist conclusion to L2 shared memory and reasoning snapshot
            self._persist_conclusion(thread)
        finally:
            # Record cooldown for all participants to prevent re-triggering
            for participant_id in thread.participants:
                self._record_discussion_participation(participant_id, status=thread.status)

            # NFR03: Force flush final state — always sync card regardless of pending state
            if on_round_complete is not None:
                final_thread = self._pending_card_update or thread
                for _attempt in range(2):  # retry max=2
                    try:
                        on_round_complete(final_thread)
                        break
                    except Exception:
                        if _attempt == 0:
                            import time as _time
                            _time.sleep(0.5)  # backoff before retry
                        else:
                            logger.warning("Final card flush failed after retry for thread %s", thread.thread_id)
                self._pending_card_update = None
            self._discussion_depth.pop(thread.thread_id, None)
            self.unbind_task(thread.thread_id)

        return thread

    def _enrich_conclusion_with_speakers(self, thread: "DiscussionThread") -> str:
        """Enrich conclusion text by replacing agent_id references with emoji+name.

        Builds a mapping from each participant's agent_id[:8] to their
        emoji+name display string, then replaces occurrences in the conclusion.
        Also prepends a participants header line.

        Args:
            thread: The completed discussion thread.

        Returns:
            Enriched conclusion text with emoji+name labels.
        """
        conclusion = thread.conclusion or ""
        if not conclusion:
            return conclusion

        # Build replacement mapping: agent_id[:8] → "emoji name"
        replacements: dict[str, str] = {}
        participant_displays: list[str] = []
        for pid in thread.participants:
            display = self._resolve_agent_display(pid)
            participant_displays.append(display)
            # Map the truncated ID (used in summarize_conclusion's input) to display
            short_id = pid[:8]
            if short_id and short_id != display:
                replacements[short_id] = display

        # Replace truncated agent_id references in the conclusion text
        enriched = conclusion
        for short_id, display in replacements.items():
            enriched = enriched.replace(short_id, display)

        # Prepend participants header
        if participant_displays:
            header = "Participants: " + ", ".join(participant_displays)
            enriched = f"{header}\n\n{enriched}"

        return enriched

    def _run_final_arbiter(self, thread: "DiscussionThread") -> None:
        """Inject a Final Arbiter instruction to force a clear conclusion.

        Called when max_rounds is exhausted without convergence. Asks the
        first participant agent to produce a definitive conclusion statement.

        If remaining token budget is insufficient (< arbiter_max_tokens),
        falls back to using the last message content as conclusion.
        """
        from src.config import get_settings

        if not thread.participants:
            return

        settings = get_settings()
        arbiter_max_tokens = settings.slock_arbiter_max_tokens

        # Check remaining budget
        remaining_budget = 0
        if thread.config:
            remaining_budget = thread.config.token_budget - thread.total_tokens_used
        if remaining_budget < arbiter_max_tokens:
            # Budget insufficient — fallback to last message as conclusion
            if thread.messages:
                fallback = thread.messages[-1].content
                if fallback and (not thread.conclusion or len(thread.conclusion.strip()) < 10):
                    thread.conclusion = fallback
            logger.info(
                "Final Arbiter skipped (budget insufficient: remaining=%d < max=%d) | thread=%s",
                remaining_budget, arbiter_max_tokens, thread.thread_id[:8],
            )
            return

        # Select arbiter: choose the participant who spoke LEAST to reduce bias.
        # The initiator (participants[0]) tends to be biased toward their original
        # framing; picking the least-spoken agent produces a more neutral arbiter.
        message_counts: dict[str, int] = {}
        for p in thread.participants:
            message_counts[p] = 0
        for msg in thread.messages:
            if msg.sender_agent_id in message_counts:
                message_counts[msg.sender_agent_id] = message_counts.get(msg.sender_agent_id, 0) + 1
        # Sort by message count ascending; break ties by position (later = less invested)
        sorted_participants = sorted(
            thread.participants,
            key=lambda p: (message_counts.get(p, 0), -thread.participants.index(p)),
        )
        arbiter_agent_id = sorted_participants[0] if sorted_participants else thread.participants[0]
        registry = getattr(getattr(self._engine, "registry", None), "list_agents", None)
        if callable(registry):
            try:
                known_ids = {getattr(agent, "agent_id", "") for agent in registry()}
            except Exception:
                known_ids = set()
            if known_ids and arbiter_agent_id not in known_ids:
                return

        # Build arbiter prompt — strict JSON output for security and parseability
        # Include recent discussion history so arbiter has full context
        recent_history = ""
        if thread.messages:
            history_msgs = thread.messages[-10:]  # Last 10 messages for context
            history_lines = []
            for msg in history_msgs:
                speaker = msg.sender_agent_id.split(":")[-1] if ":" in msg.sender_agent_id else msg.sender_agent_id
                content_preview = msg.content[:300] if msg.content else ""
                history_lines.append(f"- [{speaker}]: {content_preview}")
            recent_history = (
                "DISCUSSION HISTORY (recent messages):\n"
                + "\n".join(history_lines)
                + "\n\n"
            )

        arbiter_prompt = (
            "[SYSTEM - Final Arbiter]\n"
            "The discussion has reached its maximum rounds without consensus. "
            "You MUST now produce a clear, definitive conclusion.\n\n"
            f"{recent_history}"
            "REQUIREMENTS:\n"
            "1. Summarize the key points discussed\n"
            "2. State a firm decision or recommendation\n"
            "3. Do NOT use uncertain language (no '不确定', 'needs review', 'maybe')\n"
            "4. Consider ALL participants' viewpoints fairly\n\n"
            "OUTPUT FORMAT — Respond ONLY with valid JSON, no other text:\n"
            '{"conclusion": "完整的结论文本", "key_points": ["要点1", "要点2"], "decision": "明确决策"}\n\n'
            "All three fields are REQUIRED. Do NOT include markdown code blocks, explanations, or any text outside the JSON."
        )

        try:
            response, tokens = self._execute_agent_turn(
                arbiter_agent_id, arbiter_prompt, thread=thread,
            )
            if response and response.strip():
                raw_response = response.strip()
                # Try to parse as JSON first
                parsed = None
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError:
                    # Try to extract JSON from code blocks or surrounding text
                    json_match = re.search(r'\{[\s\S]*\}', raw_response)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            parsed = None

                if parsed and isinstance(parsed, dict):
                    # Extract conclusion with fallback to decision field
                    conclusion = parsed.get("conclusion") or parsed.get("decision") or ""
                    if not conclusion and parsed.get("key_points"):
                        conclusion = "; ".join(parsed["key_points"])

                    if conclusion:
                        thread.conclusion = conclusion.strip()
                        thread.total_tokens_used += tokens
                        logger.info(
                            "Final Arbiter: JSON parsed successfully | conclusion=%d chars | key_points=%d | thread=%s",
                            len(conclusion),
                            len(parsed.get("key_points", [])),
                            thread.thread_id[:8],
                        )
                    else:
                        # JSON parsed but no meaningful conclusion — fallback to raw
                        thread.conclusion = raw_response
                        thread.total_tokens_used += tokens
                        logger.warning(
                            "Final Arbiter: JSON parsed but empty conclusion, using raw response | thread=%s",
                            thread.thread_id[:8],
                        )
                else:
                    # JSON parse failed — fallback to raw response (backward compatible)
                    thread.conclusion = raw_response
                    thread.total_tokens_used += tokens
                    logger.warning(
                        "Final Arbiter: JSON parse failed, falling back to raw text | thread=%s",
                        thread.thread_id[:8],
                    )
            else:
                logger.warning(
                    "Final Arbiter returned empty response | thread=%s",
                    thread.thread_id[:8],
                )
        except Exception as exc:
            logger.warning(
                "Final Arbiter failed: %s | thread=%s — using existing conclusion",
                str(exc), thread.thread_id[:8],
            )

    def _escalate_to_council(self, thread: "DiscussionThread") -> None:
        """Escalate a deadlocked discussion to the Council protocol.

        Called when the Final Arbiter fails to produce a meaningful conclusion.
        The Council protocol (independent answers → anonymous review → synthesis)
        provides a more rigorous decision-making process for complex disagreements.
        """
        from .council_manager import CouncilManager

        if not thread.participants or len(thread.participants) < 2:
            return

        # Extract the discussion question from the trigger message or topic
        question = ""
        if thread.messages:
            # Use the first message as the question
            question = thread.messages[0].content[:500]
        if not question:
            question = f"讨论未能达成共识，请各位独立给出判断: {thread.topic or 'unknown topic'}"

        # Build council manager using the engine's execution protocol
        engine = self._engine
        if not hasattr(engine, '_council_manager') or engine._council_manager is None:
            logger.warning(
                "Council escalation failed: no council_manager available | thread=%s",
                thread.thread_id[:8],
            )
            return

        try:
            council_mgr: CouncilManager = engine._council_manager
            run = council_mgr.run_council(
                question=question,
                agent_ids=thread.participants[:5],  # Cap at 5 participants
                channel_id=getattr(thread, 'channel_id', ''),
            )
            if run and run.final_answer:
                thread.conclusion = f"[Council决策] {run.final_answer}"
                thread.status = DiscussionStatus.CONVERGED
                logger.info(
                    "Discussion escalated to Council successfully | thread=%s conclusion_len=%d",
                    thread.thread_id[:8], len(thread.conclusion),
                )
            else:
                logger.warning(
                    "Council escalation produced no answer | thread=%s",
                    thread.thread_id[:8],
                )
        except Exception as exc:
            logger.warning(
                "Council escalation failed: %s | thread=%s",
                str(exc), thread.thread_id[:8],
            )

    def detect_organic_divergence(self, channel_id: str, recent_outputs: list[dict]) -> bool:
        """Detect organic divergence between agents' recent outputs in the same channel.

        When multiple agents produce conflicting recommendations on the same topic
        without an explicit discussion, this detects the divergence and can
        trigger a discussion to resolve it.

        Args:
            channel_id: The channel to check for divergence.
            recent_outputs: List of dicts with keys: agent_id, content, timestamp, task_content.
                           Should contain outputs from the last N minutes.

        Returns:
            True if divergence was detected and a discussion was triggered.
        """
        if len(recent_outputs) < 2:
            return False

        # Group outputs by similar task content (agents working on related tasks)
        from collections import defaultdict
        topic_groups: dict[str, list[dict]] = defaultdict(list)
        for output in recent_outputs:
            # Use first 50 chars of task_content as rough topic key
            topic_key = (output.get("task_content") or "")[:50].strip().lower()
            if topic_key:
                topic_groups[topic_key].append(output)

        for topic_key, group in topic_groups.items():
            if len(group) < 2:
                continue

            # Simple divergence heuristic: check for contradictory signals
            # Look for opposing sentiment markers in outputs from different agents
            _POSITIVE_MARKERS = {"通过", "可以", "建议采用", "推荐", "approve", "LGTM", "looks good"}
            _NEGATIVE_MARKERS = {"不通过", "不建议", "反对", "reject", "有问题", "需要修改", "不可以"}

            positive_agents: list[str] = []
            negative_agents: list[str] = []

            for item in group:
                content = item.get("content", "")
                agent_id = item.get("agent_id", "")
                has_positive = any(m in content for m in _POSITIVE_MARKERS)
                has_negative = any(m in content for m in _NEGATIVE_MARKERS)

                if has_positive and not has_negative:
                    positive_agents.append(agent_id)
                elif has_negative and not has_positive:
                    negative_agents.append(agent_id)

            # Divergence: at least one agent positive and one negative
            if positive_agents and negative_agents:
                # Trigger a discussion between the divergent agents
                participants = list(set(positive_agents[:2] + negative_agents[:2]))
                trigger_msg = (
                    f"[系统检测到观点分歧] 关于「{topic_key[:30]}」，"
                    f"部分成员持肯定意见，部分成员持否定意见。请讨论并达成共识。"
                )
                try:
                    thread = DiscussionThread(
                        thread_id=str(uuid.uuid4()),
                        channel_id=channel_id,
                        participants=participants,
                        config=self._config,
                        trigger_reason="organic_divergence",
                        topic=self._build_thread_topic(
                            f"divergence:{topic_key[:30]}", participants[0], channel_id=channel_id
                        ),
                    )
                    self.run_discussion(thread, trigger_msg)
                    logger.info(
                        "Organic divergence detected and discussion triggered | "
                        "channel=%s topic=%s positive=%s negative=%s",
                        channel_id[:8], topic_key[:30], positive_agents, negative_agents,
                    )
                    return True
                except Exception as exc:
                    logger.warning("Failed to trigger divergence discussion: %s", repr(exc))

        return False

    def _detect_conflict_llm_semantic(self, conclusion: str, key_knowledge: str) -> tuple[bool, str]:
        """使用 LLM 进行语义冲突检测，5s 超时后降级到规则检测。

        Returns (has_conflict, conflict_details).
        On timeout or error, returns (False, "") to let caller fall back to rule-based detection.
        """
        if not conclusion or not key_knowledge or self._engine is None:
            return False, ""

        # Build prompt for LLM semantic conflict detection
        prompt = f"""你是一个知识冲突检测专家。请判断以下两段文本是否存在逻辑冲突。

【结论文本】:
{conclusion[:500]}

【关键知识】:
{key_knowledge[:500]}

请分析这两段文本是否存在逻辑冲突或语义矛盾。

如果存在冲突，请按以下 JSON 格式回答:
{{"conflict": true, "reason": "简要描述冲突点"}}

如果不存在冲突，请按以下 JSON 格式回答:
{{"conflict": false, "reason": ""}}

只返回 JSON，不要添加其他解释。"""

        try:
            def _call_llm() -> tuple[bool, str]:
                try:
                    # Use the engine's LLM session API with 5s timeout
                    # We use a dummy agent approach or direct API call
                    # Since we don't have a specific agent for this, we'll use any available agent
                    # or call the underlying model directly

                    # Try to get any agent from the engine for LLM access
                    agents = []
                    if hasattr(self._engine, 'registry'):
                        registry = self._engine.registry
                        if hasattr(registry, 'list_agents'):
                            channel_id = getattr(self._engine, 'chat_id', '') or getattr(self._engine, 'channel_id', '')
                            agents = list(registry.list_agents(channel_id=channel_id) if channel_id else registry.list_agents())

                    if not agents:
                        return False, ""

                    agent = agents[0]

                    # Call LLM with 30 second timeout (ACP startup + prompt)
                    result = self._engine.run_agent_session_full(
                        agent, prompt, timeout=30.0, max_tokens=200,
                    )

                    if result and result.text:
                        response_text = result.text.strip()
                        # Try to parse JSON response
                        import json
                        try:
                            # Extract JSON from response (handle markdown code blocks)
                            json_str = response_text
                            if "```" in response_text:
                                # Extract content between code blocks
                                import re
                                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                                if match:
                                    json_str = match.group(1)
                                else:
                                    # Try to find JSON object
                                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                                    if match:
                                        json_str = match.group(0)

                            parsed = json.loads(json_str)
                            has_conflict = bool(parsed.get("conflict", False))
                            reason = str(parsed.get("reason", "")) if has_conflict else ""
                            return has_conflict, reason
                        except (json.JSONDecodeError, ValueError):
                            # Fallback: check for conflict keywords in response
                            has_conflict = "conflict" in response_text.lower() or "冲突" in response_text
                            return has_conflict, response_text[:200] if has_conflict else ""

                    return False, ""
                except Exception as exc:
                    logger.debug("LLM semantic conflict detection failed: %s", str(exc))
                    return False, ""

            # Execute with timeout protection via daemon thread
            result_box: list[tuple[bool, str]] = []
            error_box: list[Exception] = []

            def _run() -> None:
                try:
                    result_box.append(_call_llm())
                except Exception as e:
                    error_box.append(e)

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            worker.join(timeout=35.0)
            if worker.is_alive():
                logger.warning("LLM semantic conflict detection timed out after 35s")
                return False, ""
            if error_box:
                logger.debug("LLM semantic conflict detection failed: %s", str(error_box[0]))
                return False, ""
            return result_box[0] if result_box else (False, "")

        except Exception as exc:
            logger.warning("LLM semantic conflict detection error: %s", str(exc))
            return False, ""

    def _detect_knowledge_conflict(self, conclusion: str, key_knowledge: str) -> tuple[bool, str, bool]:
        """Detect if conclusion conflicts with existing key_knowledge.

        Uses rule-based antonym pair matching first, then LLM semantic detection.
        Returns (has_conflict, conflict_details, needs_escalation).

        - needs_escalation is True when a conflict is detected (either by rules or LLM)
        - On LLM timeout/error, falls back to rule-based result only
        """
        if not conclusion or not key_knowledge:
            return False, "", False

        # Step 1: Rule-based detection
        conclusion_lower = conclusion.lower()
        kk_lower = key_knowledge.lower()

        conflicts_found: list[str] = []
        for term_a, term_b in _CONFLICT_PAIRS:
            has_a_in_conclusion = term_a in conclusion_lower
            has_b_in_conclusion = term_b in conclusion_lower
            has_a_in_kk = term_a in kk_lower
            has_b_in_kk = term_b in kk_lower

            # Conclusion says A but Key Knowledge says B
            if has_a_in_conclusion and has_b_in_kk:
                conflicts_found.append(f"结论使用 '{term_a}'，但 Key Knowledge 包含 '{term_b}'")
            # Conclusion says B but Key Knowledge says A
            if has_b_in_conclusion and has_a_in_kk:
                conflicts_found.append(f"结论使用 '{term_b}'，但 Key Knowledge 包含 '{term_a}'")

        if conflicts_found:
            # Rule-based detection found conflict - needs escalation
            return True, "; ".join(conflicts_found), True

        # Step 2: LLM semantic detection (only if rules didn't find conflict)
        has_llm_conflict, llm_details = self._detect_conflict_llm_semantic(conclusion, key_knowledge)
        if has_llm_conflict:
            # LLM detected semantic conflict - needs escalation
            return True, llm_details or "LLM 检测到语义冲突", True

        # No conflict detected
        return False, "", False

    def _persist_conclusion(self, thread: "DiscussionThread") -> None:
        """Persist discussion conclusion to L2 shared memory and task reasoning snapshot.

        Called automatically after run_discussion completes (CONVERGED/TIMEOUT/BUDGET_EXHAUSTED).
        """
        if not thread.conclusion:
            return

        # Determine channel_id from participants or engine
        channel_id = getattr(thread, "channel_id", "") or ""
        if not channel_id and self._engine:
            channel = getattr(self._engine, "chat_id", None)
            if channel:
                channel_id = channel

        memory_mgr = self._memory_manager
        if not memory_mgr:
            logger.debug("No memory_manager available, skipping conclusion persistence")
            return

        # Enrich conclusion text with emoji+name for referenced speakers (Task 10)
        enriched_conclusion = self._enrich_conclusion_with_speakers(thread)

        # Build display-friendly participant list
        participants_display = [
            self._resolve_agent_display(pid) for pid in thread.participants
        ]

        # 1. Write to L2 SHARED_MEMORY.md
        if channel_id:
            try:
                # Generate topic_hash from trigger_reason or conclusion
                hash_source = thread.trigger_reason[:30] if thread.trigger_reason else thread.conclusion[:30]
                topic_hash = hashlib.md5(hash_source.encode()).hexdigest()[:8]
                # Title: first line of conclusion, max 60 chars
                title = thread.conclusion[:60].split('\n')[0]
                memory_mgr.append_discussion_conclusion(
                    channel_id,
                    enriched_conclusion,
                    section="Discussion History",
                    topic_hash=topic_hash,
                    title=title,
                    participants=participants_display,
                )
                logger.info(
                    "Discussion conclusion persisted to L2 | thread=%s channel=%s",
                    thread.thread_id[:8], channel_id[:8],
                )
            except Exception as exc:
                logger.warning("Failed to persist conclusion to L2: %s", str(exc))

        # 2. Write to reasoning snapshot if bound to a task
        task_id = self.get_bound_task(thread.thread_id)
        if task_id and thread.participants:
            # Write conclusion to first participant's reasoning snapshot
            primary_agent_id = thread.participants[0]
            try:
                prompt_summary = f"Discussion: {thread.topic[:100] if thread.topic else 'inter-agent'}"
                result_summary = f"Discussion conclusion: {thread.conclusion[:300]}"
                existing = memory_mgr.read_agent_reasoning_snapshot(primary_agent_id, task_id)
                if existing:
                    prompt_summary = existing.get("prompt_summary", prompt_summary)
                    result_summary = f"{existing.get('result_summary', '')} | {result_summary}"
                memory_mgr.write_agent_reasoning_snapshot(
                    primary_agent_id, task_id,
                    prompt_summary=prompt_summary,
                    result_summary=result_summary,
                )
                logger.info(
                    "Discussion conclusion written to reasoning snapshot | agent=%s task=%s",
                    primary_agent_id[:8], task_id[:8],
                )
            except Exception as exc:
                logger.warning("Failed to write conclusion to reasoning snapshot: %s", str(exc))

        # 3. Sync to all participant agents' L1 (with knowledge conflict detection)
        if thread.participants and thread.conclusion:
            try:
                trigger = getattr(thread, "trigger_reason", "")
                # Check for knowledge conflicts before syncing
                agents_to_skip: set[str] = set()
                for agent_id in thread.participants:
                    memory = memory_mgr.read_agent_memory(agent_id)
                    if memory and memory.key_knowledge:
                        has_conflict, conflict_details, needs_escalation = self._detect_knowledge_conflict(
                            thread.conclusion, memory.key_knowledge
                        )
                        if has_conflict:
                            # Trigger escalation for this agent
                            logger.warning(
                                "Knowledge conflict detected for agent %s: %s",
                                agent_id[:8], conflict_details,
                            )
                            agents_to_skip.add(agent_id)

                            # Send conflict escalation card if needs_escalation is True
                            if needs_escalation and self._engine:
                                agent = self._engine.get_agent(agent_id)
                                if agent:
                                    # Try to send conflict escalation card via engine callback
                                    card_send_fn = getattr(self._engine, "conclusion_card_callback", None)
                                    if card_send_fn:
                                        from .card_templates import build_conflict_escalation_card
                                        agent_name = getattr(agent, 'display_name', '') or getattr(agent, 'name', '') or agent_id[:8]
                                        conflict_card = build_conflict_escalation_card(
                                            agent_name=agent_name,
                                            conflict_details=conflict_details,
                                            conclusion=thread.conclusion,
                                            key_knowledge=memory.key_knowledge,
                                            channel_id=channel_id,
                                            thread_id=thread.thread_id,
                                        )
                                        try:
                                            card_send_fn(conflict_card)
                                            logger.info(
                                                "Sent conflict escalation card for agent %s",
                                                agent_id[:8],
                                            )
                                        except Exception as card_exc:
                                            logger.warning(
                                                "Failed to send conflict escalation card: %s",
                                                str(card_exc),
                                            )

                                    # Also try the existing escalate_fn for backward compatibility
                                    from .models import EscalationLevel
                                    escalate_fn = getattr(self._engine, "escalate", None)
                                    if escalate_fn:
                                        escalate_fn(
                                            agent,
                                            f"讨论结论与 Key Knowledge 冲突: {conflict_details}",
                                            level=EscalationLevel.WARNING,
                                            context=f"结论: {thread.conclusion[:200]}",
                                            options=["覆盖写入", "跳过同步", "人工审查"],
                                        )

                # Sync only to agents without conflicts
                agents_to_sync = [aid for aid in thread.participants if aid not in agents_to_skip]
                if agents_to_sync:
                    memory_mgr.sync_discussion_conclusion_to_agents(
                        agents_to_sync, thread.conclusion, trigger_reason=trigger,
                    )
                if agents_to_skip:
                    logger.info(
                        "Skipped L1 sync for %d agents due to knowledge conflicts (pending human confirmation)",
                        len(agents_to_skip),
                    )
            except Exception as exc:
                logger.warning("Failed to sync conclusion to agent L1: %s", str(exc))

        # 4. Send lightweight notification card to channel
        self._send_conclusion_notification(thread, participants_display)

        # 5. Drive task state from discussion verdict (if bound to a task)
        self._apply_verdict_to_bound_task(thread)

    def _apply_verdict_to_bound_task(self, thread: "DiscussionThread") -> None:
        """If discussion is bound to a task, extract verdict and drive task state.

        APPROVE signal → complete the task via task board.
        REJECT signal → trigger rejection via orchestrator for retry.
        Neutral → no action (existing behavior: just persist to memory).
        """
        task_id = self.get_bound_task(thread.thread_id)
        if not task_id or not thread.conclusion:
            return

        verdict = self._extract_verdict_from_conclusion(thread.conclusion)
        if verdict == "neutral":
            return

        # Get engine references for task board and orchestrator
        engine = self._engine
        if not engine:
            logger.debug("No engine reference, cannot apply verdict to task %s", task_id)
            return

        task_board = getattr(engine, "_task_mgr", None)
        orchestrator = getattr(engine, "_collaboration_orchestrator", None)

        if verdict == "approve":
            if task_board:
                # Find the task's claimer and complete it
                claimer_id = ""
                tasks = getattr(task_board, "_tasks", [])
                for t in tasks:
                    if t.task_id == task_id and t.claimed_by:
                        claimer_id = t.claimed_by
                        break
                if claimer_id:
                    try:
                        task_board.complete_task(task_id, claimer_id)
                        logger.info(
                            "Discussion verdict APPROVE applied to task %s",
                            task_id,
                        )
                    except Exception as exc:
                        logger.warning("Failed to apply APPROVE verdict: %s", repr(exc))
        elif verdict == "reject":
            # Determine rejector from discussion participants
            rejector_role = ""
            if thread.participants and engine:
                # Use the last speaker as rejector
                last_participant = thread.participants[-1] if len(thread.participants) > 1 else thread.participants[0]
                agent = engine.get_agent(last_participant)
                if agent:
                    rejector_role = getattr(agent, "role", "") or "reviewer"

            if orchestrator:
                try:
                    orchestrator.on_task_rejected(task_id, rejector_role)
                    logger.info(
                        "Discussion verdict REJECT applied to task %s (rejector=%s)",
                        task_id, rejector_role,
                    )
                except Exception as exc:
                    logger.warning("Failed to apply REJECT verdict: %s", repr(exc))

    @staticmethod
    def _extract_verdict_from_conclusion(conclusion: str) -> str:
        """Extract verdict signal from discussion conclusion text.

        Returns: "approve", "reject", or "neutral".
        """
        upper = conclusion.upper()
        lower = conclusion.lower()

        # Strong approve signals
        approve_signals = [
            "[APPROVE]", "LGTM", "APPROVED",
            "通过", "合格", "没有问题", "可以发布", "确认通过",
            "同意方案", "方案可行",
        ]
        # Strong reject signals
        reject_signals = [
            "[REJECT]", "REJECTED",
            "不通过", "打回", "需要修改", "需要重做", "存在问题",
            "方案不可行", "建议重新", "质量不达标",
        ]

        for signal in reject_signals:
            if signal in upper or signal in lower or signal in conclusion:
                return "reject"

        for signal in approve_signals:
            if signal in upper or signal in lower or signal in conclusion:
                return "approve"

        return "neutral"

    def _send_conclusion_notification(
        self, thread: "DiscussionThread", participants_display: list[str]
    ) -> None:
        """Send a notification card after conclusion persistence completes."""
        if not thread.conclusion:
            return
        try:
            from .card_templates import build_conclusion_notification_card

            preview = thread.conclusion[:100]
            card = build_conclusion_notification_card(preview, participants_display)

            # Use engine's conclusion_card_callback if available
            engine = self._engine
            if engine and engine.conclusion_card_callback:
                engine.conclusion_card_callback(card)
        except Exception as exc:
            logger.debug("Failed to send conclusion notification card: %s", str(exc))

    # ------------------------------------------------------------------
    # Stop Discussion
    # ------------------------------------------------------------------

    def stop_discussion(self, thread: DiscussionThread) -> DiscussionThread:
        """Manually stop an active discussion.

        Sets status to MANUALLY_STOPPED, timestamps completion, and
        summarizes whatever conclusion is available. Persists the conclusion
        to L2 shared memory and L1 agent memories.

        Args:
            thread: The discussion thread to stop.

        Returns:
            The stopped thread with conclusion.
        """
        thread.status = DiscussionStatus.MANUALLY_STOPPED
        thread.completed_at = time.time()
        self.summarize_conclusion(thread)

        # Persist conclusion (same as run_discussion normal exit path)
        try:
            self._persist_conclusion(thread)
        except Exception as exc:
            # Fallback: if LLM summarization fails, use last message as raw conclusion
            logger.warning(
                "stop_discussion: _persist_conclusion failed: %s — using raw conclusion",
                str(exc),
            )
            if not thread.conclusion and thread.messages:
                thread.conclusion = thread.messages[-1].content

        logger.info(
            "Discussion %s manually stopped (rounds=%d, tokens=%d)",
            thread.thread_id[:8],
            thread.current_round,
            thread.total_tokens_used,
        )
        return thread

    # ------------------------------------------------------------------
    # Hint Injection (人工干预)
    # ------------------------------------------------------------------

    def inject_hint(self, thread: DiscussionThread, hint: str) -> bool:
        """Inject a user hint into an active discussion thread.

        The hint will be injected into the system prompt of the next
        discussion round, providing guidance to the agents when the
        discussion is stuck or needs direction.

        Args:
            thread: The discussion thread to inject the hint into.
            hint: The user-provided guidance text.

        Returns:
            True if hint was successfully added, False if thread is not active.
        """
        if not thread.is_active:
            logger.warning(
                "Cannot inject hint into inactive thread %s (status=%s)",
                thread.thread_id[:8],
                thread.status.value,
            )
            return False

        if not hint or not hint.strip():
            logger.warning("Empty hint rejected for thread %s", thread.thread_id[:8])
            return False

        thread.add_hint(hint.strip())
        logger.info(
            "Hint injected into discussion %s (hint_len=%d)",
            thread.thread_id[:8],
            len(hint),
        )
        return True

    # ------------------------------------------------------------------
    # Placeholder / Integration Methods
    # ------------------------------------------------------------------

    def _execute_agent_turn(self, agent_id: str, prompt: str, thread=None) -> tuple[Optional[str], int]:
        """Execute a single agent turn in the discussion.

        Uses the engine's public API to build a prompt (with memory context)
        and run an ACP session for the target agent.

        Args:
            agent_id: The ID of the agent to execute.
            prompt: The discussion prompt to send.
            thread: Optional DiscussionThread for timeout calculation and cancellation.

        Returns:
            Tuple of (response_text, output_tokens). output_tokens is the real
            completion token count from ACP usage reporting; falls back to
            len(text)//4 estimation when usage is unavailable.
        """
        use_unavailable_null = "_notify_unavailable" in getattr(self, "__dict__", {})

        if self._engine is None:
            logger.warning(
                "Discussion unavailable: no engine for agent %s",
                agent_id[:8],
            )
            self._notify_unavailable(agent_id, "engine not available")
            if use_unavailable_null:
                return None, 0
            placeholder = f"[placeholder response from {agent_id}]"
            return placeholder, self._estimate_tokens(placeholder)

        # Calculate per-turn timeout from discussion config
        turn_timeout = 60  # default fallback
        if thread and hasattr(thread, 'config') and thread.config:
            config = thread.config
            disc_timeout = getattr(config, 'discussion_timeout', None)
            max_rounds = getattr(config, 'max_rounds', None)
            if disc_timeout and max_rounds and max_rounds > 0:
                turn_timeout = disc_timeout / max_rounds
        elif self._config:
            disc_timeout = getattr(self._config, 'discussion_timeout', None)
            max_rounds = getattr(self._config, 'max_rounds', None)
            if disc_timeout and max_rounds and max_rounds > 0:
                turn_timeout = disc_timeout / max_rounds

        # Resolve max_tokens from engine settings
        max_tokens: Optional[int] = None
        if self._engine is not None:
            settings = self._engine.settings
            if settings is not None:
                max_tokens = getattr(settings, 'slock_max_tokens_per_round', None)

        try:
            # Resolve agent identity via public API
            agent = self._engine.get_agent(agent_id)
            if agent is None:
                logger.warning(
                    "Discussion unavailable: agent %s not found in registry",
                    agent_id[:8],
                )
                self._notify_unavailable(agent_id, "agent not found in registry")
                if use_unavailable_null:
                    return None, 0
                placeholder = f"[placeholder response from {agent_id}]"
                return placeholder, self._estimate_tokens(placeholder)

            # Build full prompt with system prompt, memory, and discussion context
            # via public API (memory=None triggers auto-read)
            full_prompt = self._engine.build_agent_prompt(agent, prompt)

            # Execute via full API to get token usage
            result = self._engine.run_agent_session_full(
                agent, full_prompt, timeout=turn_timeout, max_tokens=max_tokens,
            )
            if result:
                response_text = result.text or ""
                # Detect truncation: if stop_reason indicates max_tokens was hit,
                # append a visible marker so downstream consumers know the output
                # was cut short.
                if result.stop_reason in ("max_tokens", "length"):
                    response_text += "[输出因长度限制被截断]"
                    logger.info(
                        "Agent %s response truncated (stop_reason=%s, max_tokens=%s)",
                        agent_id[:8],
                        result.stop_reason,
                        max_tokens,
                    )
                # Use real output_tokens from ACP; fallback to len//4 estimate
                output_tokens = result.output_tokens if (result.output_tokens or 0) > 0 else self._estimate_tokens(response_text)
                logger.debug(
                    "Agent %s responded (length=%d, tokens=%d, timeout=%.1fs)",
                    agent_id[:8],
                    len(response_text),
                    output_tokens,
                    turn_timeout,
                )
                return response_text, output_tokens

            logger.warning(
                "ACP session returned None for agent %s; using fallback",
                agent_id[:8],
            )
            placeholder = f"[placeholder response from {agent_id}]"
            return placeholder, self._estimate_tokens(placeholder)

        except Exception as exc:
            logger.error(
                "Error executing agent turn for %s: %s",
                agent_id[:8],
                exc,
                exc_info=True,
            )
            placeholder = f"[placeholder response from {agent_id}]"
            return placeholder, self._estimate_tokens(placeholder)

    def _notify_unavailable(self, agent_id: str, reason: str) -> None:
        """Notify via callback that an agent is unavailable during discussion."""
        if self._on_unavailable_notify:
            try:
                self._on_unavailable_notify(agent_id, reason)
            except Exception as exc:
                logger.debug("on_unavailable_notify callback failed: %s", str(exc))

    def _call_llm_for_summary(self, prompt: str) -> str:
        """Call LLM to generate a discussion summary.

        Creates a temporary one-shot ACP session to run the summarization
        prompt, then closes the session. Falls back to a structured summary
        if the LLM call fails.

        Args:
            prompt: The summarization prompt.

        Returns:
            The summary text (falls back to structured format).
        """
        fallback = (
            "Discussion summary (auto-generated fallback): "
            "Participants exchanged perspectives. "
            "See thread messages for full details."
        )

        if self._engine is None:
            logger.debug("No engine available for LLM summary; using fallback.")
            return fallback

        try:
            from ..agent_session import close_session_safely, create_auxiliary_session

            agent_type = getattr(self._engine, "agent_type", "coco") or "coco"
            cwd = getattr(self._engine, "root_path", ".") or "."

            session = create_auxiliary_session(
                agent_type=agent_type,
                cwd=cwd,
                thread_id=f"slock_summary_{id(prompt) & 0xFFFF:04x}",
            )
            if session is None:
                logger.warning("Failed to create summary ACP session; using fallback.")
                return fallback

            try:
                result = session.send_prompt(prompt, timeout=60)
                if result and result.text:
                    return result.text
                logger.debug("LLM summary session returned empty; using fallback.")
                return fallback
            finally:
                close_session_safely(session)

        except Exception as exc:
            logger.error(
                "Error calling LLM for summary: %s", exc, exc_info=True
            )
            return fallback

    def _find_best_discussion_partner(
        self, agent: AgentIdentity, content: str, channel_id: str = ""
    ) -> Optional[str]:
        """Find the best discussion partner by skill profile matching.

        Strategy:
        1. Extract skill domain keywords from the content
        2. List available agents in the same channel (excluding the initiator)
        3. Score agents by skill_profile tag overlap
        4. Return the best match; fallback to any IDLE agent if no skill match
        5. Return None only if channel has no other agents
        """
        registry = self._engine.registry if self._engine else None
        if registry is None:
            return None

        # Get all agents in channel (or all if no channel filter)
        agents = registry.list_agents(channel_id=channel_id) if channel_id else registry.list_agents()
        # Exclude the initiating agent
        candidates = [a for a in agents if a.agent_id != agent.agent_id]
        if not candidates:
            return None

        # Try skill-based matching via TaskRouter if available
        task_router = getattr(self._engine, "_router", None) or getattr(
            self._engine, "_task_router", None
        )
        if task_router and hasattr(task_router, "extract_skill_keywords"):
            required_skills = task_router.extract_skill_keywords(content)
            if required_skills:
                # Score candidates by role relevance
                scored: list[tuple[AgentIdentity, int]] = []
                for candidate in candidates:
                    score = 0
                    # Match agent role against required skills
                    if candidate.role in required_skills:
                        score += 2
                    # Check if any skill keyword matches the agent's role
                    for skill in required_skills:
                        if skill in candidate.role:
                            score += 1
                    scored.append((candidate, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                if scored[0][1] > 0:
                    return scored[0][0].agent_id

        # Fallback: prefer agents with relevant roles (reviewer, then planner, then any)
        for preferred_role in ("reviewer", "planner", "architect"):
            for candidate in candidates:
                if candidate.role == preferred_role:
                    return candidate.agent_id

        # Last fallback: any available agent
        return candidates[0].agent_id if candidates else None

    def _find_agent_by_role(self, role: str, channel_id: str = "") -> Optional[str]:
        """Find an agent ID by its role using registry public API.

        Args:
            role: The role to search for (e.g. "reviewer", "coder").
            channel_id: Optional channel_id for cross-group isolation.

        Returns:
            The agent_id if found, None otherwise.
        """
        if self._engine is not None:
            try:
                registry = self._engine.registry
                if registry is not None and hasattr(registry, "list_agents"):
                    agents = registry.list_agents(channel_id=channel_id) if channel_id else registry.list_agents()
                    for agent in agents:
                        if hasattr(agent, "role") and agent.role == role:
                            return agent.agent_id
            except Exception as exc:
                logger.debug("Error finding agent by role '%s': %s", role, str(exc))
        return None

    def _find_agent_by_name(self, name: str, channel_id: str = "") -> Optional[str]:
        """Find an agent ID by its display name using registry public API.

        Args:
            name: The agent name to search for (case-insensitive).
            channel_id: Optional channel_id for cross-group isolation.

        Returns:
            The agent_id if found, None otherwise.
        """
        if self._engine is not None:
            try:
                registry = self._engine.registry
                if registry is not None:
                    # Try find_by_name first (dedicated API)
                    if hasattr(registry, "find_by_name"):
                        agent = registry.find_by_name(name, channel_id=channel_id) if channel_id else registry.find_by_name(name)
                        if agent:
                            return agent.agent_id
                    # Fallback: iterate list_agents
                    if hasattr(registry, "list_agents"):
                        agents = registry.list_agents(channel_id=channel_id) if channel_id else registry.list_agents()
                        for agent in agents:
                            agent_name = getattr(agent, "name", "")
                            if agent_name.lower() == name.lower():
                                return agent.agent_id
            except Exception as exc:
                logger.debug("Error finding agent by name '%s': %s", name, str(exc))
        return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using combined SequenceMatcher and token overlap.

        Uses a hybrid approach for robust convergence detection:
        1. difflib.SequenceMatcher for character-level similarity (good for short texts)
        2. Token/Jaccard overlap for word-level similarity (good for longer texts)

        Returns the maximum of the two scores for best coverage.

        Args:
            text1: First text segment.
            text2: Second text segment.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()

        if t1 == t2:
            return 1.0

        # Score 1: SequenceMatcher (character-level, good for paraphrasing detection)
        seq_matcher = difflib.SequenceMatcher(None, t1, t2)
        seq_score = seq_matcher.ratio()

        # Score 2: Token overlap (Jaccard similarity on word tokens)
        tokens1 = set(re.findall(r'\w+', t1, re.UNICODE))
        tokens2 = set(re.findall(r'\w+', t2, re.UNICODE))

        if tokens1 and tokens2:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            jaccard_score = len(intersection) / len(union) if union else 0.0
        else:
            # Fallback to character n-grams if no word tokens (e.g., pure CJK)
            n = 2
            grams1 = set(t1[i:i + n] for i in range(len(t1) - n + 1)) if len(t1) >= n else {t1}
            grams2 = set(t2[i:i + n] for i in range(len(t2) - n + 1)) if len(t2) >= n else {t2}
            if grams1 and grams2:
                intersection = grams1 & grams2
                union = grams1 | grams2
                jaccard_score = len(intersection) / len(union) if union else 0.0
            else:
                jaccard_score = 0.0

        # Return the maximum of the two scores
        return max(seq_score, jaccard_score)

    # ------------------------------------------------------------------
    # Discussion Persistence (Task 14)
    # ------------------------------------------------------------------

    def serialize_thread(self, thread: DiscussionThread) -> dict:
        """Serialize a DiscussionThread to a JSON-compatible dict for persistence."""
        return {
            "thread_id": thread.thread_id,
            "channel_id": thread.channel_id,
            "participants": thread.participants,
            "messages": [
                {
                    "message_id": m.message_id,
                    "sender_agent_id": m.sender_agent_id,
                    "receiver_agent_id": m.receiver_agent_id,
                    "content": m.content,
                    "round_num": m.round_num,
                    "timestamp": m.timestamp,
                    "token_count": m.token_count,
                }
                for m in thread.messages
            ],
            "status": thread.status.value,
            "trigger_reason": thread.trigger_reason,
            "topic": thread.topic,
            "conclusion": thread.conclusion,
            "total_tokens_used": thread.total_tokens_used,
            "created_at": thread.created_at,
            "completed_at": thread.completed_at,
            "pending_hints": thread.pending_hints,
        }

    def deserialize_thread(
        self, data: dict, config: Optional[DiscussionConfig] = None, *, channel_id: str = ""
    ) -> DiscussionThread:
        """Deserialize a dict back into a DiscussionThread.

        Args:
            data: Serialized thread dict.
            config: Optional config override.
            channel_id: Fallback channel_id if not present in data.
        """
        effective_config = config if config is not None else self._config
        messages = [
            DiscussionMessage(
                message_id=m.get("message_id", ""),
                sender_agent_id=m.get("sender_agent_id", ""),
                receiver_agent_id=m.get("receiver_agent_id", ""),
                content=m.get("content", ""),
                round_num=m.get("round_num", 0),
                timestamp=m.get("timestamp", 0.0),
                token_count=m.get("token_count", 0),
            )
            for m in data.get("messages", [])
        ]
        status_str = data.get("status", "active")
        try:
            status = DiscussionStatus(status_str)
        except (ValueError, KeyError):
            status = DiscussionStatus.ACTIVE

        resolved_channel_id = data.get("channel_id", "") or channel_id

        return DiscussionThread(
            thread_id=data.get("thread_id", ""),
            channel_id=resolved_channel_id,
            participants=data.get("participants", []),
            messages=messages,
            status=status,
            config=effective_config,
            trigger_reason=data.get("trigger_reason", ""),
            topic=data.get("topic", ""),
            conclusion=data.get("conclusion", ""),
            total_tokens_used=data.get("total_tokens_used", 0),
            created_at=data.get("created_at", 0.0),
            completed_at=data.get("completed_at"),
            pending_hints=data.get("pending_hints", []),
        )

    def persist_discussions(self, channel_id: str, threads: list[DiscussionThread]) -> None:
        """Persist active discussion threads for a channel via memory manager."""
        if self._memory_manager is None:
            return
        serialized = [self.serialize_thread(t) for t in threads if t.is_active]
        try:
            self._memory_manager.write_discussions(channel_id, serialized)
        except Exception as exc:
            logger.debug("Failed to persist discussions for %s: %s", channel_id, str(exc))

    def load_discussions(self, channel_id: str) -> list[DiscussionThread]:
        """Load persisted discussion threads for a channel."""
        if self._memory_manager is None:
            return []
        try:
            data_list = self._memory_manager.read_discussions(channel_id)
            return [self.deserialize_thread(d, channel_id=channel_id) for d in data_list]
        except Exception as exc:
            logger.debug("Failed to load discussions for %s: %s", channel_id, str(exc))
            return []
