"""Unit tests for slock_engine/models.py — data models and serialization."""

from __future__ import annotations

from src.slock_engine.models import (
    AgentIdentity,
    AgentStatus,
    DiscussionStatus,
    SkillProfile,
    SlockChannel,
    SlockMemory,
    SlockTask,
    TaskStatus,
)


def test_persisted_status_values_are_stable():
    assert {status.name: status.value for status in AgentStatus} == {
        "IDLE": "idle",
        "WAKING": "waking",
        "THINKING": "thinking",
        "RUNNING": "running",
        "CHECKING": "checking",
        "SENDING": "sending",
        "MOVING": "moving",
        "DISCUSSING": "discussing",
        "PENDING_DISCUSSION": "pending_discussion",
    }
    assert {status.name: status.value for status in TaskStatus} == {
        "TODO": "todo",
        "IN_PROGRESS": "in_progress",
        "IN_REVIEW": "in_review",
        "DONE": "done",
    }


class TestAgentIdentity:
    def test_display_name_with_name(self):
        a = AgentIdentity(name="Bob", emoji="🔧")
        assert a.display_name == "🔧 Bob"

    def test_display_name_without_name(self):
        a = AgentIdentity(emoji="🤖")
        assert a.display_name == "🤖 Agent"

    def test_card_color_known_role(self):
        a = AgentIdentity(role="coder")
        assert a.card_color == "blue"

    def test_card_color_unknown_role(self):
        a = AgentIdentity(role="unknown_role")
        assert a.card_color == "grey"

    def test_to_dict_round_trip(self):
        a = AgentIdentity(
            agent_id="id-1",
            name="Tester",
            emoji="✅",
            agent_type="claude",
            model_name="claude-3",
            system_prompt="You are a tester.",
            role="tester",
            permissions=["shell"],
            memory_path="/tmp/mem",
            owner_group="g1",
            created_at=1000.0,
        )
        d = a.to_dict()
        restored = AgentIdentity.from_dict(d)
        assert restored.agent_id == "id-1"
        assert restored.name == "Tester"
        assert restored.emoji == "✅"
        assert restored.agent_type == "claude"
        assert restored.model_name == "claude-3"
        assert restored.system_prompt == "You are a tester."
        assert restored.role == "tester"
        assert restored.permissions == ["shell"]
        assert restored.memory_path == "/tmp/mem"
        assert restored.owner_group == "g1"
        assert restored.created_at == 1000.0

    def test_from_dict_defaults(self):
        a = AgentIdentity.from_dict({})
        assert a.emoji == "🤖"
        assert a.agent_type == "coco"
        assert a.role == "custom"


class TestSlockTask:
    def test_to_dict_round_trip(self):
        t = SlockTask(
            task_id="t1",
            content="Build feature",
            status=TaskStatus.IN_PROGRESS,
            claimed_by="agent-1",
            claimed_at=2000.0,
            created_in="ch-1",
            created_at=1000.0,
        )
        d = t.to_dict()
        restored = SlockTask.from_dict(d)
        assert restored.task_id == "t1"
        assert restored.content == "Build feature"
        assert restored.status == TaskStatus.IN_PROGRESS
        assert restored.claimed_by == "agent-1"
        assert restored.claimed_at == 2000.0
        assert restored.created_in == "ch-1"
        assert restored.created_at == 1000.0

    def test_from_dict_defaults(self):
        t = SlockTask.from_dict({})
        assert t.status == TaskStatus.TODO
        assert t.claimed_by is None


class TestSlockChannel:
    def test_to_dict_round_trip(self):
        ch = SlockChannel(
            channel_id="c2",
            name="Team-A",
            agents=["a1", "a2"],
            shared_memory_path="/tmp/shared",
            team_name="Alpha",
            created_at=500.0,
        )
        d = ch.to_dict()
        restored = SlockChannel.from_dict(d)
        assert restored.channel_id == "c2"
        assert restored.name == "Team-A"
        assert restored.agents == ["a1", "a2"]
        assert restored.team_name == "Alpha"


class TestSlockMemory:
    def test_to_markdown_all_sections(self):
        m = SlockMemory(
            role="Backend dev",
            key_knowledge="Python expert",
            active_context="Working on auth",
        )
        md = m.to_markdown()
        assert "# Role\nBackend dev" in md
        assert "# Key Knowledge\nPython expert" in md
        assert "# Active Context\nWorking on auth" in md

    def test_to_markdown_empty(self):
        m = SlockMemory()
        assert m.to_markdown() == ""

    def test_from_markdown_round_trip(self):
        original = SlockMemory(
            role="Frontend dev",
            key_knowledge="React",
            active_context="Building UI",
        )
        md = original.to_markdown()
        restored = SlockMemory.from_markdown(md)
        assert restored.role == "Frontend dev"
        assert restored.key_knowledge == "React"
        assert restored.active_context == "Building UI"

    def test_from_markdown_empty(self):
        m = SlockMemory.from_markdown("")
        assert m.role == ""
        assert m.key_knowledge == ""
        assert m.active_context == ""

    def test_from_markdown_partial(self):
        md = "# Role\nDesigner"
        m = SlockMemory.from_markdown(md)
        assert m.role == "Designer"
        assert m.key_knowledge == ""
        assert m.active_context == ""


class TestSkillProfile:
    def test_to_dict_round_trip(self):
        sp = SkillProfile(tag="review", success_rate=80.0, total_tasks=10, last_active=999.0)
        d = sp.to_dict()
        restored = SkillProfile.from_dict(d)
        assert restored.tag == "review"
        assert restored.success_rate == 80.0
        assert restored.total_tasks == 10
        assert restored.last_active == 999.0


class TestMaxRoundsReachedStatus:
    """AC-R03: MAX_ROUNDS_REACHED 与 TIMEOUT 区分。"""

    def test_terminal_status_values_are_stable_and_unique(self):
        terminal = [
            DiscussionStatus.CONVERGED,
            DiscussionStatus.TIMEOUT,
            DiscussionStatus.MAX_ROUNDS_REACHED,
            DiscussionStatus.MANUALLY_STOPPED,
        ]
        values = [s.value for s in terminal]
        assert DiscussionStatus.MAX_ROUNDS_REACHED.value == "max_rounds_reached"
        assert len(values) == len(set(values)), "Terminal status values must be unique"


class TestSlockChannelBootstrapFailed:
    """Test SlockChannel bootstrap_failed field contract."""

    def test_missing_bootstrap_failed_defaults_and_serializes_false(self):
        channel = SlockChannel.from_dict({"channel_id": "test_123"})

        assert channel.bootstrap_failed is False
        assert channel.to_dict()["bootstrap_failed"] is False

    def test_bootstrap_failed_survives_round_trip(self):
        original = SlockChannel(
            channel_id="test_123",
            name="Test Team",
            bootstrap_failed=True,
        )
        data = original.to_dict()
        restored = SlockChannel.from_dict(data)
        assert restored.bootstrap_failed is True
        assert restored.channel_id == original.channel_id
        assert restored.name == original.name
