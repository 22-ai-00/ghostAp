"""Contract tests for ACP protocol model structural invariants.

These tests act as guardrails for future simplification of src/acp/models.py and
related ACP infrastructure. They assert structural invariants (field existence,
type correctness, enum stability, serialization round-trip) without pinning
specific runtime values.

Protected regression scenarios:
- ACPEventType enum must have exactly 7 members with stable string values
- ACPSessionState.to_dict() must produce a dict with exactly 7 specified keys
- ACPSessionState.from_dict(to_dict()) round-trip must preserve all fields
- ToolCallInfo.kind must accept the 8 documented values
- PromptResult must have the documented field set
"""

from __future__ import annotations

from dataclasses import fields

from src.acp.models import (
    ACPEvent,
    ACPEventType,
    ACPSessionState,
    PlanEntryInfo,
    PlanInfo,
    PromptResult,
    ToolCallInfo,
)

# ---------------------------------------------------------------------------
# ACPEventType enum stability
# ---------------------------------------------------------------------------

_EXPECTED_EVENT_TYPES = {
    "TEXT_CHUNK": "text_chunk",
    "THOUGHT_CHUNK": "thought_chunk",
    "IMAGE_CHUNK": "image_chunk",
    "TOOL_CALL_START": "tool_call_start",
    "TOOL_CALL_UPDATE": "tool_call_update",
    "TOOL_CALL_DONE": "tool_call_done",
    "PLAN_UPDATE": "plan_update",
}


class TestACPEventTypeStability:
    """Assert ACPEventType enum membership and values are stable."""

    def test_member_count(self) -> None:
        """ACPEventType must have exactly 7 members."""
        assert len(ACPEventType) == 7

    def test_member_names_and_values(self) -> None:
        """Each member name must map to the expected string value."""
        actual = {member.name: member.value for member in ACPEventType}
        assert actual == _EXPECTED_EVENT_TYPES

    def test_all_values_are_strings(self) -> None:
        """All ACPEventType values must be strings."""
        for member in ACPEventType:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# ACPSessionState serialization contract
# ---------------------------------------------------------------------------

_SESSION_STATE_DICT_KEYS = {
    "session_id",
    "agent_type",
    "cwd",
    "created_at",
    "message_count",
    "is_active",
    "last_active",
}


class TestACPSessionStateContract:
    """Assert ACPSessionState serialization invariants."""

    def _make_state(self) -> ACPSessionState:
        return ACPSessionState(
            session_id="test-session-001",
            agent_type="coco",
            cwd="/tmp/test",
            created_at=1700000000.0,
            message_count=5,
            is_active=True,
            last_active=1700000100.0,
        )

    def test_to_dict_keys(self) -> None:
        """to_dict() must produce exactly the documented key set."""
        state = self._make_state()
        d = state.to_dict()
        assert set(d.keys()) == _SESSION_STATE_DICT_KEYS

    def test_to_dict_value_types(self) -> None:
        """to_dict() values must have correct types."""
        state = self._make_state()
        d = state.to_dict()
        assert isinstance(d["session_id"], str)
        assert isinstance(d["agent_type"], str)
        assert isinstance(d["cwd"], str)
        assert isinstance(d["created_at"], float)
        assert isinstance(d["message_count"], int)
        assert isinstance(d["is_active"], bool)
        assert isinstance(d["last_active"], float)

    def test_round_trip_preserves_fields(self) -> None:
        """from_dict(to_dict(state)) must produce an equal state."""
        original = self._make_state()
        restored = ACPSessionState.from_dict(original.to_dict())
        assert restored.session_id == original.session_id
        assert restored.agent_type == original.agent_type
        assert restored.cwd == original.cwd
        assert restored.created_at == original.created_at
        assert restored.message_count == original.message_count
        assert restored.is_active == original.is_active
        assert restored.last_active == original.last_active

    def test_from_dict_with_missing_optional_fields(self) -> None:
        """from_dict() must handle missing optional fields with defaults."""
        minimal = {
            "session_id": "s1",
            "agent_type": "claude",
            "cwd": "/tmp",
        }
        state = ACPSessionState.from_dict(minimal)
        assert state.session_id == "s1"
        assert state.agent_type == "claude"
        assert state.cwd == "/tmp"
        assert isinstance(state.created_at, float)
        assert state.message_count == 0
        assert state.is_active is True
        assert isinstance(state.last_active, float)


# ---------------------------------------------------------------------------
# ToolCallInfo structural contract
# ---------------------------------------------------------------------------

_VALID_TOOL_CALL_KINDS = {"read", "edit", "delete", "execute", "think", "search", "fetch", "other"}
_VALID_TOOL_CALL_STATUSES = {"pending", "in_progress", "completed", "failed"}


class TestToolCallInfoContract:
    """Assert ToolCallInfo field existence and kind/status value sets."""

    def test_required_fields_exist(self) -> None:
        """ToolCallInfo must have id, title, kind, status, content, locations, result fields."""
        field_names = {f.name for f in fields(ToolCallInfo)}
        expected = {"id", "title", "kind", "status", "content", "locations", "result"}
        assert expected <= field_names

    def test_kind_values_are_documented(self) -> None:
        """All documented kind values must be accepted without error."""
        for kind in _VALID_TOOL_CALL_KINDS:
            info = ToolCallInfo(id="t1", title="test", kind=kind, status="pending")
            assert info.kind == kind

    def test_status_values_are_documented(self) -> None:
        """All documented status values must be accepted without error."""
        for status in _VALID_TOOL_CALL_STATUSES:
            info = ToolCallInfo(id="t1", title="test", kind="other", status=status)
            assert info.status == status

    def test_default_fields(self) -> None:
        """Default values for optional fields must be correct types."""
        info = ToolCallInfo(id="t1", title="test", kind="read", status="pending")
        assert info.content == ""
        assert isinstance(info.locations, list)
        assert info.result is None


# ---------------------------------------------------------------------------
# PromptResult structural contract
# ---------------------------------------------------------------------------

_VALID_STOP_REASONS = {"end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"}


class TestPromptResultContract:
    """Assert PromptResult field existence and structural invariants."""

    def test_required_fields_exist(self) -> None:
        """PromptResult must have the documented field set."""
        field_names = {f.name for f in fields(PromptResult)}
        expected = {"stop_reason", "text", "tool_calls", "tool_results", "plan", "modified_files"}
        assert expected <= field_names

    def test_stop_reason_values(self) -> None:
        """All documented stop_reason values must be accepted."""
        for reason in _VALID_STOP_REASONS:
            result = PromptResult(stop_reason=reason)
            assert result.stop_reason == reason

    def test_default_fields(self) -> None:
        """Default values must match documented types."""
        result = PromptResult(stop_reason="end_turn")
        assert result.text == ""
        assert isinstance(result.tool_calls, list)
        assert isinstance(result.tool_results, list)
        assert result.plan is None
        assert isinstance(result.modified_files, set)


# ---------------------------------------------------------------------------
# ACPEvent structural contract
# ---------------------------------------------------------------------------


class TestACPEventContract:
    """Assert ACPEvent field existence and type constraints."""

    def test_required_fields_exist(self) -> None:
        """ACPEvent must have the documented field set."""
        field_names = {f.name for f in fields(ACPEvent)}
        expected = {
            "event_type",
            "text",
            "tool_call",
            "plan",
            "source_id",
            "message_id",
            "codex_message_phase",
            "timestamp",
        }
        assert expected <= field_names

    def test_minimal_construction(self) -> None:
        """ACPEvent can be constructed with only event_type."""
        event = ACPEvent(event_type=ACPEventType.TEXT_CHUNK)
        assert event.event_type == ACPEventType.TEXT_CHUNK
        assert event.text is None
        assert event.tool_call is None
        assert event.plan is None
        assert event.source_id is None
        assert event.message_id is None
        assert event.codex_message_phase is None
        assert isinstance(event.timestamp, float)


# ---------------------------------------------------------------------------
# PlanEntryInfo / PlanInfo structural contract
# ---------------------------------------------------------------------------


class TestPlanInfoContract:
    """Assert PlanEntryInfo and PlanInfo structural invariants."""

    def test_plan_entry_fields(self) -> None:
        """PlanEntryInfo must have content, priority, status fields."""
        field_names = {f.name for f in fields(PlanEntryInfo)}
        assert {"content", "priority", "status"} <= field_names

    def test_plan_entry_defaults(self) -> None:
        """PlanEntryInfo defaults must be correct."""
        entry = PlanEntryInfo(content="test task")
        assert entry.priority == "medium"
        assert entry.status == "pending"

    def test_plan_info_entries_field(self) -> None:
        """PlanInfo must have entries list field."""
        field_names = {f.name for f in fields(PlanInfo)}
        assert "entries" in field_names
        plan = PlanInfo()
        assert isinstance(plan.entries, list)
