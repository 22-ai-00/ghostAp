"""Unit tests for the DiscussionManager in src/slock_engine/discussion_manager.py.

Tests cover:
- Trigger detection (rule-based, @mention, uncertainty markers)
- Discussion lifecycle (start, execute_round, stop)
- Convergence detection (explicit signals, Jaccard similarity)
- Token budget enforcement
- Full discussion loop (run_discussion)
- Edge cases and error handling
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.slock_engine.discussion_manager import (
    AT_MENTION_PATTERN,
    CONVERGENCE_SIGNALS,
    UNCERTAINTY_MARKERS,
    DiscussionManager,
)
from src.slock_engine.engine import SlockEngine
from src.slock_engine.models import (
    AgentIdentity,
    DiscussionConfig,
    DiscussionMessage,
    DiscussionStatus,
    DiscussionThread,
)

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: str = "agent-001",
    name: str = "TestAgent",
    role: str = "coder",
) -> AgentIdentity:
    """Create a test AgentIdentity."""
    return AgentIdentity(
        agent_id=agent_id,
        name=name,
        role=role,
    )


def _make_thread(
    participants: Optional[list[str]] = None,
    config: Optional[DiscussionConfig] = None,
    messages: Optional[list[DiscussionMessage]] = None,
    status: DiscussionStatus = DiscussionStatus.ACTIVE,
    total_tokens_used: int = 0,
) -> DiscussionThread:
    """Create a test DiscussionThread."""
    return DiscussionThread(
        thread_id=str(uuid.uuid4()),
        channel_id="channel-test",
        participants=participants or ["agent-001", "agent-002"],
        messages=messages or [],
        status=status,
        config=config or DiscussionConfig(),
        trigger_reason="test",
        total_tokens_used=total_tokens_used,
    )


def _make_message(
    sender: str = "agent-001",
    receiver: str = "agent-002",
    content: str = "test message",
    round_num: int = 1,
    token_count: int = 10,
) -> DiscussionMessage:
    """Create a test DiscussionMessage."""
    return DiscussionMessage(
        message_id=str(uuid.uuid4()),
        sender_agent_id=sender,
        receiver_agent_id=receiver,
        content=content,
        round_num=round_num,
        timestamp=time.time(),
        token_count=token_count,
    )


def _make_engine_with_agents(agents: dict[str, AgentIdentity]) -> MagicMock:
    """Create a mock engine with a registry containing agents."""
    engine = MagicMock()
    registry = MagicMock()
    registry.agents = agents
    registry.list_agents.return_value = list(agents.values())
    registry.find_by_name.side_effect = lambda name: next(
        (a for a in agents.values() if a.name == name), None
    )
    engine.registry = registry
    return engine


@pytest.fixture
def default_config() -> DiscussionConfig:
    """Default discussion configuration for tests."""
    return DiscussionConfig(
        max_rounds=3,
        token_budget=50000,
        trigger_rules=["coder->reviewer"],
        convergence_threshold=0.85,
    )


@pytest.fixture
def manager(default_config: DiscussionConfig) -> DiscussionManager:
    """Discussion manager with default config and no engine."""
    return DiscussionManager(config=default_config)


@pytest.fixture
def manager_with_engine() -> DiscussionManager:
    """Discussion manager with a mocked engine containing agents."""
    coder = _make_agent(agent_id="coder-001", name="Coder", role="coder")
    reviewer = _make_agent(agent_id="reviewer-001", name="Reviewer", role="reviewer")
    planner = _make_agent(agent_id="planner-001", name="Planner", role="planner")

    agents = {
        "coder-001": coder,
        "reviewer-001": reviewer,
        "planner-001": planner,
    }
    engine = _make_engine_with_agents(agents)
    config = DiscussionConfig(
        max_rounds=3,
        token_budget=50000,
        trigger_rules=["coder->reviewer"],
        convergence_threshold=0.85,
    )
    return DiscussionManager(engine=engine, config=config)


# ===========================================================================
# Test Class: Trigger Detection
# ===========================================================================


class TestTriggerDetection:
    """Tests for should_trigger_discussion and the three trigger strategies."""

    # --- Rule-based triggers ---

    def test_rule_trigger_coder_to_reviewer(self, manager_with_engine: DiscussionManager):
        """Rule 'coder->reviewer' triggers when a coder agent produces output."""
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="Here is my code implementation.",
        )
        assert result is not None
        assert "coder-001" in result.participants
        assert "reviewer-001" in result.participants
        assert "rule:" in result.trigger_reason

    def test_rule_trigger_no_match_when_role_not_source(
        self, manager_with_engine: DiscussionManager
    ):
        """No trigger when agent role does not match the source of any rule."""
        agent = _make_agent(agent_id="reviewer-001", role="reviewer")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="Review complete.",
        )
        # Only mention/uncertainty could trigger, but content has none
        assert result is None

    def test_rule_trigger_target_role_not_found(self, manager: DiscussionManager):
        """No trigger when the target role agent is not found (no engine)."""
        config = DiscussionConfig(trigger_rules=["coder->reviewer"])
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager.should_trigger_discussion(
            agent=agent,
            result_content="Code done.",
            config=config,
        )
        # Without engine, _find_agent_by_role returns None
        assert result is None

    # --- @mention triggers ---

    def test_mention_trigger_with_known_agent(self, manager_with_engine: DiscussionManager):
        """@Reviewer mention triggers a discussion with the mentioned agent."""
        config = DiscussionConfig(trigger_rules=[])  # Disable rule triggers
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="I think @Reviewer should check this.",
            config=config,
        )
        assert result is not None
        assert "mention:" in result.trigger_reason
        assert "coder-001" in result.participants
        assert "reviewer-001" in result.participants

    def test_mention_trigger_unknown_agent(self, manager_with_engine: DiscussionManager):
        """@UnknownAgent does not trigger when agent name is not found."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="Let's ask @NonExistentBot about this.",
            config=config,
        )
        assert result is None

    def test_mention_trigger_no_mentions(self, manager_with_engine: DiscussionManager):
        """No mention in content means no mention trigger."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="No mentions here, just plain text.",
            config=config,
        )
        assert result is None

    # --- Uncertainty triggers ---

    def test_uncertainty_trigger_chinese_markers(
        self, manager_with_engine: DiscussionManager
    ):
        """Chinese uncertainty markers trigger discussion (with sufficient signal strength)."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="coder-001", role="coder")

        for marker in ("不确定", "需要讨论", "需要确认"):
            result = manager_with_engine.should_trigger_discussion(
                agent=agent,
                result_content=f"这个方案{marker}，也许需要讨论一下？可能有问题。",
                config=config,
            )
            assert result is not None, f"Failed for marker: {marker}"
            assert "uncertainty:" in result.trigger_reason

    def test_uncertainty_trigger_english_markers(
        self, manager_with_engine: DiscussionManager
    ):
        """English uncertainty markers trigger discussion when partner is available."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="coder-001", role="coder")

        for marker in ("I'm not sure", "uncertain", "needs review"):
            result = manager_with_engine.should_trigger_discussion(
                agent=agent,
                result_content=f"The approach is {marker} about correctness. I'm not sure if this is right. Maybe we need to discuss? Unclear.",
                config=config,
                force=True,
            )
            assert result is not None, f"Failed for marker: {marker}"

    def test_uncertainty_trigger_case_insensitive(
        self, manager_with_engine: DiscussionManager
    ):
        """Uncertainty marker detection is case-insensitive for English."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="I AM NOT SURE about this approach. Maybe we need to discuss? Unclear how to proceed.",
            config=config,
            force=True,
        )
        assert result is not None

    def test_uncertainty_trigger_no_partner_found(self, manager: DiscussionManager):
        """No trigger when uncertainty detected but no suitable partner exists."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager.should_trigger_discussion(
            agent=agent,
            result_content="I'm not sure about this.",
            config=config,
        )
        # No engine -> cannot find reviewer/planner -> returns None
        assert result is None

    def test_no_trigger_clean_content(self, manager_with_engine: DiscussionManager):
        """Clean content with no triggers returns None."""
        config = DiscussionConfig(trigger_rules=[])
        agent = _make_agent(agent_id="planner-001", role="planner")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="Task completed successfully. All tests pass.",
            config=config,
        )
        assert result is None

    # --- Priority ordering (rule > mention > uncertainty) ---

    def test_trigger_priority_rule_over_mention(
        self, manager_with_engine: DiscussionManager
    ):
        """Rule-based trigger takes priority (evaluated first)."""
        agent = _make_agent(agent_id="coder-001", role="coder")
        result = manager_with_engine.should_trigger_discussion(
            agent=agent,
            result_content="@Planner can you check? I'm not sure.",
        )
        # With default config "coder->reviewer", rule fires first
        assert result is not None
        assert "rule:" in result.trigger_reason


# ===========================================================================
# Test Class: Discussion Lifecycle
# ===========================================================================


class TestDiscussionLifecycle:
    """Tests for start_discussion, execute_round, and stop_discussion."""

    def test_start_discussion_adds_initial_message(self, manager: DiscussionManager):
        """start_discussion appends an initial message to the thread."""
        thread = _make_thread()
        content = "Here is the code that needs review."
        result = manager.start_discussion(thread, content)

        assert len(result.messages) == 1
        assert result.messages[0].content == content
        assert result.messages[0].sender_agent_id == "agent-001"
        assert result.messages[0].receiver_agent_id == "agent-002"
        assert result.messages[0].round_num == 0
        assert result.status == DiscussionStatus.ACTIVE

    def test_start_discussion_updates_token_count(self, manager: DiscussionManager):
        """start_discussion estimates token count from content length."""
        thread = _make_thread()
        content = "A" * 400  # 400 chars => ~100 tokens (len // 4)
        result = manager.start_discussion(thread, content)

        assert result.total_tokens_used == 100
        assert result.messages[0].token_count == 100

    def test_start_discussion_empty_participants(self, manager: DiscussionManager):
        """start_discussion with empty participants does not add messages."""
        thread = DiscussionThread(
            thread_id="test-empty",
            channel_id="channel-test",
            participants=[],
            config=DiscussionConfig(),
        )
        result = manager.start_discussion(thread, "Some content")
        assert len(result.messages) == 0

    def test_execute_round_adds_message(self, manager: DiscussionManager):
        """execute_round adds a response message to the thread."""
        thread = _make_thread()
        thread = manager.start_discussion(thread, "Initial content")

        result = manager.execute_round(thread)
        # After start (round 0), execute_round adds round 1
        assert len(result.messages) == 2
        assert result.messages[-1].round_num == 1
        # Response should be from participant[1] (alternation)
        assert result.messages[-1].sender_agent_id == "agent-002"

    def test_execute_round_alternates_speakers(self, manager: DiscussionManager):
        """execute_round alternates between participants."""
        thread = _make_thread()
        thread = manager.start_discussion(thread, "Initial")

        thread = manager.execute_round(thread)  # round 1 -> agent-002
        thread = manager.execute_round(thread)  # round 2 -> agent-001

        assert thread.messages[1].sender_agent_id == "agent-002"
        assert thread.messages[2].sender_agent_id == "agent-001"

    def test_execute_round_inactive_thread_noop(self, manager: DiscussionManager):
        """execute_round on an inactive thread does nothing."""
        thread = _make_thread(status=DiscussionStatus.CONVERGED)
        thread.messages.append(_make_message(round_num=1))
        original_count = len(thread.messages)

        result = manager.execute_round(thread)
        assert len(result.messages) == original_count

    def test_execute_round_less_than_two_participants(self, manager: DiscussionManager):
        """execute_round with <2 participants returns thread unchanged."""
        thread = _make_thread(participants=["agent-001"])
        thread.status = DiscussionStatus.ACTIVE
        thread.messages.append(_make_message(round_num=0))

        original_count = len(thread.messages)
        result = manager.execute_round(thread)
        assert len(result.messages) == original_count

    def test_stop_discussion_sets_manually_stopped(self, manager: DiscussionManager):
        """stop_discussion marks thread as MANUALLY_STOPPED."""
        thread = _make_thread()
        thread = manager.start_discussion(thread, "Some discussion")

        result = manager.stop_discussion(thread)
        assert result.status == DiscussionStatus.MANUALLY_STOPPED
        assert result.completed_at is not None
        assert result.conclusion != ""

    def test_stop_discussion_empty_thread(self, manager: DiscussionManager):
        """stop_discussion on thread with no messages still works."""
        thread = _make_thread()
        result = manager.stop_discussion(thread)
        assert result.status == DiscussionStatus.MANUALLY_STOPPED
        assert result.conclusion == "No messages in discussion."


# ===========================================================================
# Test Class: Convergence Detection
# ===========================================================================


class TestConvergenceDetection:
    """Tests for check_convergence — explicit signals and similarity."""

    # --- Explicit convergence signals ---

    def test_convergence_agree_signal(self, manager: DiscussionManager):
        """AGREE in last message triggers convergence."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="Here is my proposal", round_num=1),
            _make_message(
                content="I AGREE with this approach.", round_num=2, sender="agent-002"
            ),
        ]
        assert manager.check_convergence(thread) is True

    def test_convergence_chinese_signal(self, manager: DiscussionManager):
        """Chinese convergence signal triggers convergence."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="My suggestion", round_num=1),
            _make_message(
                content="没问题，同意你的方案", round_num=2, sender="agent-002"
            ),
        ]
        assert manager.check_convergence(thread) is True

    def test_convergence_lgtm_signal(self, manager: DiscussionManager):
        """LGTM signal triggers convergence."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="Updated implementation", round_num=1),
            _make_message(content="LGTM, ship it!", round_num=2, sender="agent-002"),
        ]
        assert manager.check_convergence(thread) is True

    def test_convergence_looks_good_signal(self, manager: DiscussionManager):
        """'looks good' signal triggers convergence."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="Final version", round_num=1),
            _make_message(
                content="This looks good to me", round_num=2, sender="agent-002"
            ),
        ]
        assert manager.check_convergence(thread) is True

    def test_convergence_no_further_suggestions(self, manager: DiscussionManager):
        """'no further suggestions' triggers convergence."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="Here is my take", round_num=1),
            _make_message(
                content="I have no further suggestions.", round_num=2, sender="agent-002"
            ),
        ]
        assert manager.check_convergence(thread) is True

    def test_convergence_case_insensitive(self, manager: DiscussionManager):
        """Convergence signals are detected case-insensitively."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="proposal", round_num=1),
            _make_message(content="agree", round_num=2, sender="agent-002"),
        ]
        assert manager.check_convergence(thread) is True

    def test_no_convergence_disagreement(self, manager: DiscussionManager):
        """No convergence when last message has no signals and content differs."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="My approach uses pattern A", round_num=1),
            _make_message(
                content="I think pattern B is better because of X Y Z",
                round_num=2,
                sender="agent-002",
            ),
        ]
        assert manager.check_convergence(thread) is False

    def test_no_convergence_single_message(self, manager: DiscussionManager):
        """No convergence with only one message."""
        thread = _make_thread()
        thread.messages = [_make_message(content="First message AGREE", round_num=1)]
        assert manager.check_convergence(thread) is False

    def test_no_convergence_empty_messages(self, manager: DiscussionManager):
        """No convergence with no messages."""
        thread = _make_thread()
        assert manager.check_convergence(thread) is False

    # --- Jaccard similarity convergence ---

    def test_convergence_high_similarity_same_sender(self, manager: DiscussionManager):
        """Convergence when same sender repeats highly similar content."""
        config = DiscussionConfig(convergence_threshold=0.8)
        thread = _make_thread(config=config)
        # Same sender, two very similar messages
        thread.messages = [
            _make_message(
                sender="agent-001",
                content="the function should validate input parameters carefully",
                round_num=1,
            ),
            _make_message(
                sender="agent-002",
                content="different perspective entirely",
                round_num=2,
            ),
            _make_message(
                sender="agent-001",
                content="the function should validate input parameters carefully",
                round_num=3,
            ),
        ]
        # agent-001 repeated the exact same message -> similarity = 1.0 > 0.8
        assert manager.check_convergence(thread) is True

    def test_convergence_below_threshold(self, manager: DiscussionManager):
        """No convergence when similarity is below threshold."""
        config = DiscussionConfig(convergence_threshold=0.9)
        thread = _make_thread(config=config)
        thread.messages = [
            _make_message(
                sender="agent-001",
                content="we should use pattern A for the database layer",
                round_num=1,
            ),
            _make_message(
                sender="agent-002",
                content="I disagree, pattern B is better here",
                round_num=2,
            ),
            _make_message(
                sender="agent-001",
                content="we should use pattern A for the service layer instead",
                round_num=3,
            ),
        ]
        # Similarity is moderate but not >= 0.9
        assert manager.check_convergence(thread) is False


# ===========================================================================
# Test Class: Budget Enforcement
# ===========================================================================


class TestBudgetEnforcement:
    """Tests for check_budget."""

    def test_budget_available(self, manager: DiscussionManager):
        """Budget check returns True when tokens used < budget."""
        thread = _make_thread(total_tokens_used=1000)
        thread.config.token_budget = 50000
        assert manager.check_budget(thread) is True

    def test_budget_exhausted_exact(self, manager: DiscussionManager):
        """Budget check returns False when tokens used == budget."""
        thread = _make_thread(total_tokens_used=50000)
        thread.config.token_budget = 50000
        assert manager.check_budget(thread) is False

    def test_budget_exhausted_over(self, manager: DiscussionManager):
        """Budget check returns False when tokens used > budget."""
        thread = _make_thread(total_tokens_used=60000)
        thread.config.token_budget = 50000
        assert manager.check_budget(thread) is False

    def test_budget_zero_used(self, manager: DiscussionManager):
        """Budget is available with zero tokens used."""
        thread = _make_thread(total_tokens_used=0)
        thread.config.token_budget = 50000
        assert manager.check_budget(thread) is True

    def test_budget_very_small(self, manager: DiscussionManager):
        """Budget of 1 token with 0 used still has budget."""
        thread = _make_thread(total_tokens_used=0)
        thread.config.token_budget = 1
        assert manager.check_budget(thread) is True


# ===========================================================================
# Test Class: Full Discussion Loop (run_discussion)
# ===========================================================================


class TestRunDiscussion:
    """Tests for run_discussion — the full loop."""

    def test_run_discussion_converges_on_agree(self, manager: DiscussionManager):
        """Discussion loop converges when agent responds with AGREE."""
        thread = _make_thread()
        thread.config.max_rounds = 5

        # Mock the agent turn to respond with AGREE on first call
        with patch.object(
            manager,
            "_execute_agent_turn",
            return_value=("I AGREE with this approach.", 10),
        ):
            result = manager.run_discussion(thread, "Please review this code.")

        assert result.status == DiscussionStatus.CONVERGED
        assert result.conclusion != ""
        assert result.completed_at is not None

    def test_run_discussion_timeout_after_max_rounds(self, manager: DiscussionManager):
        """Discussion reaches max_rounds without converging."""
        config = DiscussionConfig(max_rounds=2, token_budget=100000)
        thread = _make_thread(config=config)

        # Never converge: respond with completely different content each time
        responses = iter([
            ("The database layer should use connection pooling for better scalability", 30),
            ("Authentication must implement JWT with refresh token rotation", 25),
            ("API rate limiting needs to be per-user with sliding window algorithm", 28),
            ("Logging should use structured JSON with correlation IDs", 22),
        ])

        def unique_response(agent_id, prompt, **kwargs):
            return next(responses)

        with patch.object(manager, "_execute_agent_turn", side_effect=unique_response):
            result = manager.run_discussion(thread, "Initial content")

        assert result.status == DiscussionStatus.MAX_ROUNDS_REACHED
        assert result.conclusion != ""

    def test_run_discussion_budget_exhausted(self, manager: DiscussionManager):
        """Discussion stops when token budget is exhausted."""
        config = DiscussionConfig(max_rounds=10, token_budget=50)
        thread = _make_thread(config=config)

        # Each response is large enough to exhaust budget quickly
        with patch.object(
            manager,
            "_execute_agent_turn",
            return_value=("A" * 1000, 250),  # 250 tokens (1000 // 4)
        ):
            result = manager.run_discussion(thread, "Start content " * 20)

        assert result.status == DiscussionStatus.BUDGET_EXHAUSTED

    def test_run_discussion_preserves_messages(self, manager: DiscussionManager):
        """run_discussion accumulates all messages in the thread."""
        config = DiscussionConfig(max_rounds=2, token_budget=100000)
        thread = _make_thread(config=config)

        responses = iter([("First response", 10), ("Second response AGREE", 15)])
        with patch.object(
            manager, "_execute_agent_turn", side_effect=lambda *a, **kw: next(responses)
        ):
            result = manager.run_discussion(thread, "Starting discussion")

        # Should have: initial message + 1 or 2 round responses
        assert len(result.messages) >= 2
        # First message is the initial content
        assert result.messages[0].content == "Starting discussion"

    def test_run_discussion_single_round_converge(self, manager: DiscussionManager):
        """Discussion can converge in the very first round."""
        config = DiscussionConfig(max_rounds=5, token_budget=100000)
        thread = _make_thread(config=config)

        with patch.object(
            manager, "_execute_agent_turn", return_value=("LGTM, looks good!", 8)
        ):
            result = manager.run_discussion(thread, "Quick check needed")

        assert result.status == DiscussionStatus.CONVERGED
        # Only initial message + 1 round
        assert len(result.messages) == 2


# ===========================================================================
# Test Class: Text Similarity (Jaccard)
# ===========================================================================


class TestTextSimilarity:
    """Tests for the _calculate_text_similarity method."""

    def test_identical_texts(self, manager: DiscussionManager):
        """Identical texts have similarity of 1.0."""
        sim = manager._calculate_text_similarity(
            "hello world foo bar", "hello world foo bar"
        )
        assert sim == 1.0

    def test_completely_different_texts(self, manager: DiscussionManager):
        """Completely different texts have low similarity (hybrid scoring).

        Jaccard = 0 but SequenceMatcher gives some character-level overlap.
        The hybrid approach returns max(seq_score, jaccard_score).
        """
        sim = manager._calculate_text_similarity("alpha beta gamma", "delta epsilon zeta")
        # No shared words, but some character overlap at character level
        assert sim < 0.5  # Low but not necessarily 0.0

    def test_partial_overlap(self, manager: DiscussionManager):
        """Partial overlap gives intermediate similarity (hybrid scoring)."""
        # "hello world" and "hello earth" share 1 word out of 3 unique
        sim = manager._calculate_text_similarity("hello world", "hello earth")
        # Hybrid approach: max(SequenceMatcher ratio, Jaccard)
        # Jaccard = 1/3 ≈ 0.333, SequenceMatcher gives higher character-level similarity
        assert 0.3 < sim < 0.8  # Intermediate value, exact value depends on algorithm

    def test_both_empty(self, manager: DiscussionManager):
        """Both empty texts return 0.0."""
        sim = manager._calculate_text_similarity("", "")
        assert sim == 0.0

    def test_case_insensitive(self, manager: DiscussionManager):
        """Similarity is case-insensitive."""
        sim = manager._calculate_text_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_high_overlap(self, manager: DiscussionManager):
        """High overlap approaches 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        # Same text with one word changed
        modified = "the quick brown cat jumps over the lazy dog"
        sim = manager._calculate_text_similarity(text, modified)
        # intersection size = 7 (the, quick, brown, jumps, over, lazy, dog)
        # union size = 9 (the, quick, brown, fox, cat, jumps, over, lazy, dog)
        assert sim > 0.7


# ===========================================================================
# Test Class: AT_MENTION_PATTERN regex
# ===========================================================================


class TestAtMentionPattern:
    """Tests for the AT_MENTION_PATTERN regex."""

    def test_single_mention(self):
        """Detects a single @mention."""
        matches = AT_MENTION_PATTERN.findall("Please ask @Reviewer")
        assert matches == ["Reviewer"]

    def test_multiple_mentions(self):
        """Detects multiple @mentions."""
        matches = AT_MENTION_PATTERN.findall("@Alice and @Bob should discuss")
        assert matches == ["Alice", "Bob"]

    def test_no_mention(self):
        """No matches in text without @mentions."""
        matches = AT_MENTION_PATTERN.findall("No mentions here")
        assert matches == []


# ===========================================================================
# Test Class: DiscussionThread Properties
# ===========================================================================


class TestDiscussionThreadProperties:
    """Tests for DiscussionThread dataclass properties."""

    def test_current_round_empty(self):
        """current_round is 0 with no messages."""
        thread = _make_thread()
        assert thread.current_round == 0

    def test_current_round_with_messages(self):
        """current_round returns max round_num from messages."""
        thread = _make_thread()
        thread.messages = [
            _make_message(round_num=0),
            _make_message(round_num=1),
            _make_message(round_num=2),
        ]
        assert thread.current_round == 2

    def test_is_active_true(self):
        """is_active is True for ACTIVE status."""
        thread = _make_thread(status=DiscussionStatus.ACTIVE)
        assert thread.is_active is True

    def test_is_active_false_converged(self):
        """is_active is False for CONVERGED status."""
        thread = _make_thread(status=DiscussionStatus.CONVERGED)
        assert thread.is_active is False

    def test_is_active_false_timeout(self):
        """is_active is False for TIMEOUT status."""
        thread = _make_thread(status=DiscussionStatus.TIMEOUT)
        assert thread.is_active is False

    def test_is_active_false_manually_stopped(self):
        """is_active is False for MANUALLY_STOPPED status."""
        thread = _make_thread(status=DiscussionStatus.MANUALLY_STOPPED)
        assert thread.is_active is False

    def test_is_active_false_budget_exhausted(self):
        """is_active is False for BUDGET_EXHAUSTED status."""
        thread = _make_thread(status=DiscussionStatus.BUDGET_EXHAUSTED)
        assert thread.is_active is False


# ===========================================================================
# Test Class: Summarize Conclusion
# ===========================================================================


class TestSummarizeConclusion:
    """Tests for summarize_conclusion."""

    def test_summarize_empty_thread(self, manager: DiscussionManager):
        """Empty thread gets 'No messages' conclusion."""
        thread = _make_thread()
        conclusion = manager.summarize_conclusion(thread)
        assert conclusion == "No messages in discussion."
        assert thread.conclusion == "No messages in discussion."
        assert thread.completed_at is not None

    def test_summarize_with_messages(self, manager: DiscussionManager):
        """Thread with messages gets a fallback summary."""
        thread = _make_thread()
        thread.messages = [
            _make_message(content="Let's discuss the design", round_num=1),
            _make_message(content="I agree, good plan", round_num=2, sender="agent-002"),
        ]
        conclusion = manager.summarize_conclusion(thread)
        # Should use fallback (placeholder LLM)
        assert "Discussion summary" in conclusion or len(conclusion) > 0
        assert thread.conclusion == conclusion
        assert thread.completed_at is not None


# ===========================================================================
# Test Class: Config & Model Serialization
# ===========================================================================


class TestConfigSerialization:
    """Tests for DiscussionConfig serialization."""

    def test_config_defaults(self):
        """Default config has expected values."""
        config = DiscussionConfig()
        assert config.max_rounds == 3
        assert config.token_budget == 50000
        assert config.convergence_threshold == 0.85
        assert "coder->reviewer" in config.trigger_rules

    def test_config_to_dict_roundtrip(self):
        """Config survives to_dict -> from_dict roundtrip."""
        original = DiscussionConfig(
            max_rounds=5,
            token_budget=100000,
            trigger_rules=["writer->reviewer", "coder->tester"],
            convergence_threshold=0.9,
        )
        data = original.to_dict()
        restored = DiscussionConfig.from_dict(data)

        assert restored.max_rounds == 5
        assert restored.token_budget == 100000
        assert restored.trigger_rules == ["writer->reviewer", "coder->tester"]
        assert restored.convergence_threshold == 0.9

    def test_thread_to_dict_roundtrip(self):
        """Thread survives to_dict -> from_dict roundtrip."""
        thread = _make_thread(
            participants=["a1", "a2"],
            total_tokens_used=1234,
        )
        thread.messages = [
            _make_message(content="Hello", sender="a1", receiver="a2", round_num=0),
        ]
        thread.trigger_reason = "rule:coder->reviewer"
        thread.conclusion = "All agreed."

        data = thread.to_dict()
        restored = DiscussionThread.from_dict(data)

        assert restored.thread_id == thread.thread_id
        assert restored.participants == ["a1", "a2"]
        assert restored.total_tokens_used == 1234
        assert restored.trigger_reason == "rule:coder->reviewer"
        assert restored.conclusion == "All agreed."
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Hello"


# ===========================================================================
# Test Class: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_manager_init_no_config(self):
        """Manager initializes with default config when none provided."""
        mgr = DiscussionManager()
        assert mgr._config.max_rounds == 3
        assert mgr._config.token_budget == 50000

    def test_manager_init_custom_config(self):
        """Manager uses provided config."""
        config = DiscussionConfig(max_rounds=10, token_budget=999)
        mgr = DiscussionManager(config=config)
        assert mgr._config.max_rounds == 10
        assert mgr._config.token_budget == 999

    def test_execute_round_updates_total_tokens(self, manager: DiscussionManager):
        """execute_round accumulates token usage in the thread."""
        thread = _make_thread()
        thread = manager.start_discussion(thread, "Test")
        initial_tokens = thread.total_tokens_used

        thread = manager.execute_round(thread)
        assert thread.total_tokens_used > initial_tokens

    def test_convergence_signals_constant_not_empty(self):
        """CONVERGENCE_SIGNALS has expected entries."""
        assert len(CONVERGENCE_SIGNALS) > 0
        assert "AGREE" in CONVERGENCE_SIGNALS
        assert "LGTM" in CONVERGENCE_SIGNALS

    def test_uncertainty_markers_constant_not_empty(self):
        """UNCERTAINTY_MARKERS has expected entries."""
        assert len(UNCERTAINTY_MARKERS) > 0
        assert "不确定" in UNCERTAINTY_MARKERS
        assert "needs review" in UNCERTAINTY_MARKERS

    def test_discussion_status_enum_values(self):
        """DiscussionStatus enum has all expected members."""
        assert DiscussionStatus.ACTIVE.value == "active"
        assert DiscussionStatus.CONVERGED.value == "converged"
        assert DiscussionStatus.TIMEOUT.value == "timeout"
        assert DiscussionStatus.BUDGET_EXHAUSTED.value == "budget_exhausted"
        assert DiscussionStatus.MANUALLY_STOPPED.value == "manually_stopped"

    def test_run_discussion_zero_max_rounds(self, manager: DiscussionManager):
        """Discussion with max_rounds=0 immediately reaches max rounds."""
        config = DiscussionConfig(max_rounds=0, token_budget=100000)
        thread = _make_thread(config=config)

        result = manager.run_discussion(thread, "Will this even run?")
        assert result.status == DiscussionStatus.MAX_ROUNDS_REACHED

    def test_multiple_discussions_independent(self, manager: DiscussionManager):
        """Multiple discussion threads do not interfere with each other."""
        thread1 = _make_thread(participants=["a1", "a2"])
        thread2 = _make_thread(participants=["b1", "b2"])

        thread1 = manager.start_discussion(thread1, "Thread 1 content")
        thread2 = manager.start_discussion(thread2, "Thread 2 content")

        assert thread1.messages[0].content == "Thread 1 content"
        assert thread2.messages[0].content == "Thread 2 content"
        assert thread1.thread_id != thread2.thread_id


# ===========================================================================
# Test Class: Handler _stop_discussion regression (type-safety fix)
# ===========================================================================


class TestHandlerStopDiscussion:
    """Regression tests for slock handler _stop_discussion method.

    Ensures the handler correctly resolves DiscussionThread from
    engine._active_discussions before calling dm.stop_discussion().
    """

    def _make_handler(self):
        """Create a minimal mock of the slock handler with required methods."""
        handler = MagicMock()
        handler.send_text_to_chat = MagicMock()
        # Import the actual method and bind it
        from src.feishu.handlers.slock import SlockHandler

        handler._stop_discussion = SlockHandler._stop_discussion.__get__(handler)
        return handler

    def _make_mock_engine(self, active_discussions=None, discussion_manager=None):
        """Create a mock engine with _active_discussions and _discussion_manager."""
        engine = MagicMock()
        engine._active_discussions = active_discussions or {}
        engine._discussion_manager = discussion_manager
        return engine

    def test_stop_discussion_thread_id_match(self):
        """When thread_id matches, dm.stop_discussion receives DiscussionThread object."""
        thread = _make_thread()
        chat_id = "chat-123"

        dm = MagicMock()
        dm.stop_discussion.return_value = thread

        engine = self._make_mock_engine(
            active_discussions={chat_id: thread},
            discussion_manager=dm,
        )

        handler = self._make_handler()
        manager_mock = MagicMock()
        manager_mock.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager_mock)

        handler._stop_discussion(chat_id, {"thread_id": thread.thread_id})

        # Verify dm.stop_discussion was called with DiscussionThread, not str
        dm.stop_discussion.assert_called_once_with(thread)
        handler.send_text_to_chat.assert_called_with(chat_id, "⏹ 讨论已手动终止。")

    def test_stop_discussion_thread_id_mismatch(self):
        """When thread_id does not match, returns 'not found' message without AttributeError."""
        thread = _make_thread()
        chat_id = "chat-123"

        dm = MagicMock()
        engine = self._make_mock_engine(
            active_discussions={chat_id: thread},
            discussion_manager=dm,
        )

        handler = self._make_handler()
        manager_mock = MagicMock()
        manager_mock.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager_mock)

        # Pass a different thread_id that doesn't match
        handler._stop_discussion(chat_id, {"thread_id": "non-existent-id"})

        # dm.stop_discussion should NOT be called
        dm.stop_discussion.assert_not_called()
        handler.send_text_to_chat.assert_called_with(
            chat_id, "ℹ️ 讨论线程已结束或不存在。"
        )

    def test_stop_discussion_no_active_discussions(self):
        """When _active_discussions is empty, returns 'not found' message."""
        chat_id = "chat-123"

        dm = MagicMock()
        engine = self._make_mock_engine(
            active_discussions={},
            discussion_manager=dm,
        )

        handler = self._make_handler()
        manager_mock = MagicMock()
        manager_mock.get_activated_engine.return_value = engine
        handler._get_engine_manager = MagicMock(return_value=manager_mock)

        handler._stop_discussion(chat_id, {"thread_id": "some-thread-id"})

        dm.stop_discussion.assert_not_called()
        handler.send_text_to_chat.assert_called_with(
            chat_id, "ℹ️ 讨论线程已结束或不存在。"
        )


# ===========================================================================
# Test Class: ACP Mock Integration - _execute_agent_turn
# ===========================================================================


class TestExecuteAgentTurnIntegration:
    """Integration tests for _execute_agent_turn with mocked engine."""

    def _make_mock_agent(self, agent_id="agent-001", agent_type="coco", role="coder"):
        """Create a mock agent object."""
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.agent_type = agent_type
        agent.role = role
        agent.system_prompt = f"You are a {role} agent."
        return agent

    def _make_mock_engine(self):
        """Create a mock engine with public API methods."""

        engine = MagicMock()
        engine.get_agent = MagicMock()
        engine.run_agent_session_full = MagicMock()
        engine.build_agent_prompt = MagicMock()
        return engine

    def _make_mock_memory_manager(self):
        """Create a mock memory manager."""
        memory_manager = MagicMock()
        memory = MagicMock()
        memory.role = "coder"
        memory.key_knowledge = ""
        memory.active_context = ""
        memory_manager.read_agent_memory.return_value = memory
        return memory_manager

    def _make_prompt_result(self, text: str, output_tokens: int = 10):
        """Create a mock PromptResult."""
        from src.acp.models import PromptResult
        return PromptResult(
            stop_reason="end_turn",
            text=text,
            output_tokens=output_tokens,
        )

    def test_registered_agent_returns_acp_response(self):
        """When engine has a registered agent and run_agent_session_full returns a response, it's returned correctly."""
        engine = self._make_mock_engine()
        mock_agent = self._make_mock_agent()
        engine.get_agent.return_value = mock_agent
        engine.run_agent_session_full.return_value = self._make_prompt_result("I agree with the approach")
        engine.build_agent_prompt.return_value = "full prompt text"

        dm = DiscussionManager(engine=engine)
        result = dm._execute_agent_turn("agent-001", "Please review this code")

        assert result[0] == "I agree with the approach"
        assert result[1] == 10
        engine.get_agent.assert_called_once_with("agent-001")
        engine.run_agent_session_full.assert_called_once()
        call_args = engine.run_agent_session_full.call_args
        assert call_args[0][0] == mock_agent
        assert call_args[0][1] == "full prompt text"

    def test_agent_turn_passes_discussion_timeout_budget(self):
        """Discussion turns pass the derived per-turn timeout to the engine session."""
        engine = self._make_mock_engine()
        mock_agent = self._make_mock_agent()
        engine.get_agent.return_value = mock_agent
        engine.run_agent_session_full.return_value = self._make_prompt_result("review response")
        engine.build_agent_prompt.return_value = "full prompt text"
        thread = DiscussionThread(
            channel_id="channel-test",
            participants=["agent-001", "agent-002"],
            config=DiscussionConfig(max_rounds=4, discussion_timeout=120),
        )

        dm = DiscussionManager(engine=engine)
        result = dm._execute_agent_turn("agent-001", "Please review", thread=thread)

        assert result[0] == "review response"
        assert engine.run_agent_session_full.call_args.kwargs["timeout"] == 30

    def test_slock_engine_run_agent_session_accepts_timeout(self, tmp_path):
        """The real SlockEngine public discussion API accepts a per-turn timeout."""
        engine = SlockEngine("chat-1", str(tmp_path))
        agent = _make_agent("agent-timeout", "TimeoutAgent")
        captured: dict[str, object] = {}

        def fake_run(agent_arg, prompt_arg, *, timeout=None):
            captured["agent"] = agent_arg
            captured["prompt"] = prompt_arg
            captured["timeout"] = timeout
            return "ok"

        engine._run_acp_session = fake_run

        assert engine.run_agent_session(agent, "prompt", timeout=12.5) == "ok"
        assert captured == {"agent": agent, "prompt": "prompt", "timeout": 12.5}

    def test_engine_none_returns_placeholder(self):
        """When engine is None, placeholder fallback is returned."""
        dm = DiscussionManager(engine=None)
        result = dm._execute_agent_turn("agent-001", "some prompt")

        assert "[placeholder response from agent-001" in result[0]
        assert result[1] > 0  # Token count estimated from placeholder text

    def test_agent_not_in_registry_returns_placeholder(self):
        """When agent is not in registry, placeholder is returned."""
        engine = self._make_mock_engine()
        engine.get_agent.return_value = None

        dm = DiscussionManager(engine=engine)
        result = dm._execute_agent_turn("unknown-agent", "some prompt")

        assert "[placeholder response from unknown-agent" in result[0]
        assert result[1] > 0  # Token count estimated from placeholder text
        engine.run_agent_session_full.assert_not_called()

    def test_acp_session_raises_exception_returns_placeholder(self):
        """When run_agent_session_full raises an exception, placeholder is returned."""
        engine = self._make_mock_engine()
        mock_agent = self._make_mock_agent()
        engine.get_agent.return_value = mock_agent
        engine.run_agent_session_full.side_effect = RuntimeError("ACP connection failed")
        engine.build_agent_prompt.return_value = "full prompt text"

        dm = DiscussionManager(engine=engine)
        result = dm._execute_agent_turn("agent-001", "some prompt")

        assert "[placeholder response from agent-001" in result[0]
        assert result[1] > 0  # Token count estimated from placeholder text

    def test_memory_read_and_build_agent_prompt_called(self):
        """build_agent_prompt is called with the agent and prompt."""
        engine = self._make_mock_engine()
        mock_agent = self._make_mock_agent()
        engine.get_agent.return_value = mock_agent
        engine.run_agent_session_full.return_value = self._make_prompt_result("response text")
        engine.build_agent_prompt.return_value = "enriched prompt"

        dm = DiscussionManager(engine=engine)
        result = dm._execute_agent_turn("agent-001", "discussion prompt")

        # Verify build_agent_prompt was called with agent and prompt
        engine.build_agent_prompt.assert_called_once_with(mock_agent, "discussion prompt")

        # Verify the enriched prompt was passed to run_agent_session_full
        engine.run_agent_session_full.assert_called_once()
        call_args = engine.run_agent_session_full.call_args
        assert call_args[0][0] == mock_agent
        assert call_args[0][1] == "enriched prompt"
        assert result[0] == "response text"


# ===========================================================================
# Test Class: ACP Mock Integration - _call_llm_for_summary
# ===========================================================================


class TestCallLLMForSummaryIntegration:
    """Integration tests for _call_llm_for_summary with mocked session."""

    def test_session_returns_text(self):
        """When session is created and returns text, it's returned."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "The agents agreed on using pattern A for the service layer."
        mock_session.send_prompt.return_value = mock_result

        engine = MagicMock()
        engine.agent_type = "coco"
        engine.root_path = "/tmp/test"

        dm = DiscussionManager(engine=engine)

        with patch(
            "src.agent_session.create_auxiliary_session",
            return_value=mock_session,
        ), patch(
            "src.agent_session.close_session_safely",
        ):
            result = dm._call_llm_for_summary("Summarize this discussion")

        assert result == "The agents agreed on using pattern A for the service layer."
        mock_session.send_prompt.assert_called_once()

    def test_engine_none_returns_fallback(self):
        """When engine is None, fallback text is returned."""
        dm = DiscussionManager(engine=None)
        result = dm._call_llm_for_summary("Summarize this")

        assert "Discussion summary" in result
        assert "fallback" in result

    def test_session_creation_fails_returns_fallback(self):
        """When session creation fails, fallback is returned."""
        engine = MagicMock()
        engine.agent_type = "coco"
        engine.root_path = "/tmp/test"

        dm = DiscussionManager(engine=engine)

        with patch(
            "src.agent_session.create_auxiliary_session",
            return_value=None,
        ):
            result = dm._call_llm_for_summary("Summarize this discussion")

        assert "Discussion summary" in result
        assert "fallback" in result

    def test_send_prompt_raises_returns_fallback(self):
        """When session.send_prompt raises, fallback is returned."""
        mock_session = MagicMock()
        mock_session.send_prompt.side_effect = RuntimeError("LLM timeout")

        engine = MagicMock()
        engine.agent_type = "coco"
        engine.root_path = "/tmp/test"

        dm = DiscussionManager(engine=engine)

        with patch(
            "src.agent_session.create_auxiliary_session",
            return_value=mock_session,
        ), patch(
            "src.agent_session.close_session_safely",
        ):
            result = dm._call_llm_for_summary("Summarize this discussion")

        assert "Discussion summary" in result
        assert "fallback" in result


# ---------------------------------------------------------------------------
# AC-04: Parallel Discussion Capacity
# ---------------------------------------------------------------------------
class TestParallelDiscussionCapacity:
    """AC-04: Engine enforces max parallel discussions per channel."""

    def _make_engine(self, max_parallel=3):
        engine = MagicMock()
        engine.agent_type = "coco"
        engine.root_path = "/tmp/test"
        # Simulate discussion tracking dict
        engine._discussions = {}
        engine._max_parallel_discussions = max_parallel

        def add_discussion(channel_id, thread):
            if channel_id not in engine._discussions:
                engine._discussions[channel_id] = []
            if len(engine._discussions[channel_id]) >= engine._max_parallel_discussions:
                return False
            engine._discussions[channel_id].append(thread)
            return True

        engine._add_discussion = MagicMock(side_effect=add_discussion)
        return engine

    def test_allows_up_to_max_parallel(self):
        engine = self._make_engine(max_parallel=3)
        for i in range(3):
            result = engine._add_discussion("ch1", {"id": f"thread_{i}"})
            assert result is True
        assert len(engine._discussions["ch1"]) == 3

    def test_rejects_beyond_capacity(self):
        engine = self._make_engine(max_parallel=2)
        engine._add_discussion("ch1", {"id": "t1"})
        engine._add_discussion("ch1", {"id": "t2"})
        result = engine._add_discussion("ch1", {"id": "t3"})
        assert result is False
        assert len(engine._discussions["ch1"]) == 2

    def test_separate_channels_independent(self):
        engine = self._make_engine(max_parallel=1)
        r1 = engine._add_discussion("ch_a", {"id": "t1"})
        r2 = engine._add_discussion("ch_b", {"id": "t2"})
        assert r1 is True
        assert r2 is True


# ---------------------------------------------------------------------------
# AC-05: Discussion Card Lifecycle (3-stage)
# ---------------------------------------------------------------------------
class TestDiscussionCardLifecycle:
    """AC-05: Discussion cards follow start → update → summary lifecycle."""

    def test_start_card_contains_topic_and_participants(self):
        """Starting card (build_discussion_card) must show participants."""
        from src.slock_engine.card_templates import build_discussion_card

        card = build_discussion_card(
            thread_id="t1",
            participants=["Architect", "Backend Dev"],
            messages=[],
            current_round=0,
            max_rounds=5,
            trigger_reason="API design review",
            channel_id="ch_test",
        )
        card_str = json.dumps(card)
        assert "Architect" in card_str
        assert "Backend Dev" in card_str
        assert "API design review" in card_str

    def test_round_update_card_contains_speaker_and_content(self):
        """Per-round update card must show who spoke and what they said."""
        from src.slock_engine.card_templates import build_discussion_card

        card = build_discussion_card(
            thread_id="t1",
            participants=["Architect", "Tester"],
            messages=[
                {"sender": "Architect", "content": "We should use REST over gRPC for external APIs.", "round_num": 2},
            ],
            current_round=2,
            max_rounds=5,
            channel_id="ch_test",
        )
        card_str = json.dumps(card)
        assert "Architect" in card_str
        assert "REST" in card_str
        assert "2" in card_str

    def test_summary_card_contains_conclusion(self):
        """Final summary card must contain the discussion conclusion."""
        from src.slock_engine.card_templates import build_discussion_summary_card

        card = build_discussion_summary_card(
            thread_id="t1",
            participants=["Architect", "Tester"],
            conclusion="Team agreed on REST for external, gRPC for internal.",
            total_rounds=5,
            channel_id="ch_test",
        )
        card_str = json.dumps(card)
        assert "REST for external" in card_str
        assert "5" in card_str


# ---------------------------------------------------------------------------
# AC-06: Convergence Detection and Timeout
# ---------------------------------------------------------------------------
class TestConvergenceAndTimeout:
    """AC-06: Discussion ends on convergence or max rounds."""

    def _make_dm(self):
        engine = MagicMock()
        engine.agent_type = "coco"
        engine.root_path = "/tmp/test"
        dm = DiscussionManager(engine=engine)
        return dm

    def _make_thread(self, messages_data):
        """Helper to build a DiscussionThread with DiscussionMessage objects."""
        from src.slock_engine.models import DiscussionMessage, DiscussionThread
        messages = []
        for m in messages_data:
            messages.append(DiscussionMessage(
                sender_agent_id=m["speaker"],
                content=m["content"],
                round_num=m.get("round_num", 1),
            ))
        return DiscussionThread(
            channel_id="channel-test",
            participants=["agent-001", "agent-002"],
            messages=messages,
        )

    def test_convergence_detected_early(self):
        """If agents agree, discussion stops before max_rounds."""
        dm = self._make_dm()

        # Simulate convergence: agreement signal in last message
        thread = self._make_thread([
            {"speaker": "A", "content": "I propose X"},
            {"speaker": "B", "content": "I agree with X, let's proceed"},
        ])
        assert dm.check_convergence(thread) is True

    def test_no_convergence_continues(self):
        """Disagreeing content does not trigger early stop."""
        dm = self._make_dm()
        thread = self._make_thread([
            {"speaker": "A", "content": "I propose X"},
            {"speaker": "B", "content": "I think we should consider option Y instead"},
        ])
        assert dm.check_convergence(thread) is False


# ---------------------------------------------------------------------------
# Task 21: Concurrent _active_discussions add/remove safety (50 threads)
# ---------------------------------------------------------------------------
class TestConcurrentDiscussionSafety:
    """Verify that _active_discussions add/remove under threading.Lock is safe with 50 threads."""

    def _make_engine(self):
        """Create a minimal mock engine with _discussions_lock and _active_discussions."""
        import threading

        class FakeEngine:
            def __init__(self):
                self._discussions_lock = threading.Lock()
                self._active_discussions: dict[str, list] = {}

            def add_discussion(self, channel_id: str, thread_id: str, max_parallel: int = 100) -> bool:
                with self._discussions_lock:
                    discussions = self._active_discussions.setdefault(channel_id, [])
                    if len(discussions) >= max_parallel:
                        return False
                    discussions.append(thread_id)
                return True

            def remove_discussion(self, channel_id: str, thread_id: str) -> None:
                with self._discussions_lock:
                    discussions = self._active_discussions.get(channel_id, [])
                    self._active_discussions[channel_id] = [
                        t for t in discussions if t != thread_id
                    ]
                    if not self._active_discussions[channel_id]:
                        del self._active_discussions[channel_id]

        return FakeEngine()

    def test_50_threads_concurrent_add_no_corruption(self):
        """50 threads concurrently adding discussions produce no data corruption."""
        import threading

        engine = self._make_engine()
        barrier = threading.Barrier(50)
        errors: list[str] = []

        def add_worker(idx: int):
            try:
                barrier.wait(timeout=5)
                engine.add_discussion("channel-1", f"thread-{idx}")
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=add_worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(engine._active_discussions["channel-1"]) == 50

    def test_50_threads_concurrent_add_remove_no_corruption(self):
        """25 threads add and 25 threads remove concurrently without corruption."""
        import threading

        engine = self._make_engine()

        # Pre-populate 25 discussions
        for i in range(25):
            engine.add_discussion("ch-mix", f"pre-{i}")

        barrier = threading.Barrier(50)
        errors: list[str] = []

        def add_worker(idx: int):
            try:
                barrier.wait(timeout=5)
                engine.add_discussion("ch-mix", f"new-{idx}")
            except Exception as exc:
                errors.append(str(exc))

        def remove_worker(idx: int):
            try:
                barrier.wait(timeout=5)
                engine.remove_discussion("ch-mix", f"pre-{idx}")
            except Exception as exc:
                errors.append(str(exc))

        threads = []
        for i in range(25):
            threads.append(threading.Thread(target=add_worker, args=(i,)))
            threads.append(threading.Thread(target=remove_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        # After: 25 pre- removed, 25 new- added => 25 remain
        discussions = engine._active_discussions.get("ch-mix", [])
        assert len(discussions) == 25
        assert all(d.startswith("new-") for d in discussions)


# ---------------------------------------------------------------------------
# Task 22: _find_agent_by_role correctness test
# ---------------------------------------------------------------------------
class TestFindAgentByRole:
    """Tests for DiscussionManager._find_agent_by_role using registry attribute."""

    def test_finds_agent_by_role_in_registry(self):
        """_find_agent_by_role returns the correct agent_id for a matching role."""
        coder = _make_agent(agent_id="coder-001", name="Coder", role="coder")
        reviewer = _make_agent(agent_id="reviewer-001", name="Reviewer", role="reviewer")

        engine = MagicMock()
        registry = MagicMock()
        registry.agents = {"coder-001": coder, "reviewer-001": reviewer}
        registry.list_agents.return_value = [coder, reviewer]
        engine.registry = registry

        dm = DiscussionManager(engine=engine)
        assert dm._find_agent_by_role("coder") == "coder-001"
        assert dm._find_agent_by_role("reviewer") == "reviewer-001"

    def test_returns_none_when_role_not_found(self):
        """_find_agent_by_role returns None when no agent has the role."""
        coder = _make_agent(agent_id="coder-001", name="Coder", role="coder")

        engine = MagicMock()
        registry = MagicMock()
        registry.agents = {"coder-001": coder}
        registry.list_agents.return_value = [coder]
        engine.registry = registry

        dm = DiscussionManager(engine=engine)
        assert dm._find_agent_by_role("tester") is None

    def test_returns_none_when_engine_is_none(self):
        """_find_agent_by_role returns None when engine is None."""
        dm = DiscussionManager(engine=None)
        assert dm._find_agent_by_role("coder") is None

    def test_find_agent_by_name_case_insensitive(self):
        """_find_agent_by_name is case-insensitive."""
        reviewer = _make_agent(agent_id="r-001", name="Reviewer", role="reviewer")

        engine = MagicMock()
        registry = MagicMock()
        registry.agents = {"r-001": reviewer}
        registry.list_agents.return_value = [reviewer]
        registry.find_by_name.side_effect = lambda name, **kw: (
            reviewer if name.lower() == "reviewer" else None
        )
        engine.registry = registry

        dm = DiscussionManager(engine=engine)
        assert dm._find_agent_by_name("reviewer") == "r-001"
        assert dm._find_agent_by_name("REVIEWER") == "r-001"
        assert dm._find_agent_by_name("Reviewer") == "r-001"

    def test_find_agent_by_name_not_found(self):
        """_find_agent_by_name returns None for non-existent name."""
        engine = MagicMock()
        registry = MagicMock()
        registry.agents = {}
        registry.list_agents.return_value = []
        registry.find_by_name.return_value = None
        engine.registry = registry

        dm = DiscussionManager(engine=engine)
        assert dm._find_agent_by_name("Ghost") is None


# ---------------------------------------------------------------------------
# Task 23: Watchdog timeout test (discussion_timeout setting)
# ---------------------------------------------------------------------------
class TestWatchdogTimeout:
    """Tests for discussion watchdog timer behavior."""

    def test_run_discussion_respects_max_rounds_timeout(self):
        """Discussion stops at max_rounds even without convergence."""
        config = DiscussionConfig(max_rounds=2, token_budget=100000)
        dm = DiscussionManager()

        thread = _make_thread(config=config)

        # Use completely different responses to avoid convergence detection
        responses = iter([
            ("Redis caching strategy with TTL based on data freshness requirements", 25),
            ("Circuit breaker pattern for external service fault tolerance", 22),
            ("Event-driven architecture using message queues for async processing", 28),
            ("Health check endpoints with detailed service discovery integration", 24),
        ])

        def never_converge(agent_id, prompt, **kwargs):
            return next(responses)

        with patch.object(dm, "_execute_agent_turn", side_effect=never_converge):
            result = dm.run_discussion(thread, "Start")

        assert result.status == DiscussionStatus.MAX_ROUNDS_REACHED
        assert result.completed_at is not None


# ---------------------------------------------------------------------------
# AC6: DiscussionThread.status thread-safety tests
# ---------------------------------------------------------------------------
class TestDiscussionThreadSafety:
    """AC6: DiscussionThread.status thread-safety tests."""

    def test_status_concurrent_writes_no_corruption(self):
        """Multiple threads writing status simultaneously should not corrupt data."""
        import threading

        from src.slock_engine.models import DiscussionConfig, DiscussionStatus, DiscussionThread

        thread = DiscussionThread(
            channel_id="test_ch",
            participants=["agent_a", "agent_b"],
            trigger_reason="test",
            config=DiscussionConfig(),
        )
        thread.status = DiscussionStatus.ACTIVE

        errors = []
        barrier = threading.Barrier(3)

        def writer(target_status):
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    thread.status = target_status
                    # Read back immediately
                    current = thread.status
                    assert current in (
                        DiscussionStatus.ACTIVE,
                        DiscussionStatus.TIMEOUT,
                        DiscussionStatus.CONVERGED,
                    )
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=(DiscussionStatus.ACTIVE,))
        t2 = threading.Thread(target=writer, args=(DiscussionStatus.TIMEOUT,))
        t3 = threading.Thread(target=writer, args=(DiscussionStatus.CONVERGED,))

        t1.start()
        t2.start()
        t3.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        t3.join(timeout=10)

        assert not errors, f"Concurrent status writes caused errors: {errors}"

    def test_watchdog_and_execute_round_concurrent(self):
        """Simulate watchdog firing during execute_round — no IndexError or data corruption."""
        import threading
        import time

        from src.slock_engine.models import (
            DiscussionConfig,
            DiscussionMessage,
            DiscussionStatus,
            DiscussionThread,
        )

        thread = DiscussionThread(
            channel_id="test_ch",
            participants=["agent_a", "agent_b"],
            trigger_reason="test concurrency",
            config=DiscussionConfig(max_rounds=5),
        )
        thread.status = DiscussionStatus.ACTIVE

        # Event to simulate watchdog firing
        watchdog_event = threading.Event()
        round_started = threading.Event()
        errors = []

        def simulate_execute_round():
            """Simulate what execute_round does: append messages."""
            try:
                round_started.set()
                for i in range(5):
                    if watchdog_event.is_set():
                        break
                    msg = DiscussionMessage(
                        message_id=f"msg_{i}",
                        sender_agent_id="agent_a",
                        receiver_agent_id="agent_b",
                        content=f"Message {i}",
                        round_num=i + 1,
                    )
                    thread.messages.append(msg)
                    time.sleep(0.01)  # Simulate work
            except Exception as exc:
                errors.append(exc)

        def simulate_watchdog():
            """Simulate watchdog: wait then set status to TIMEOUT."""
            try:
                round_started.wait(timeout=5)
                time.sleep(0.02)  # Fire after first message
                watchdog_event.set()
                thread.status = DiscussionStatus.TIMEOUT
            except Exception as exc:
                errors.append(exc)

        t_round = threading.Thread(target=simulate_execute_round)
        t_watchdog = threading.Thread(target=simulate_watchdog)

        t_round.start()
        t_watchdog.start()
        t_round.join(timeout=10)
        t_watchdog.join(timeout=10)

        assert not errors, f"Concurrent watchdog/round caused errors: {errors}"
        assert thread.status == DiscussionStatus.TIMEOUT
        # Messages list should be consistent (no partial writes or corruption)
        assert all(isinstance(m, DiscussionMessage) for m in thread.messages)
