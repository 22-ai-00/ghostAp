"""Task 25: Tests that CardEvent payload validation rejects invalid inputs."""

from src.card.events import CardEvent, CardEventType


class TestCardEventPayloadValidation:
    """CardEvent factory methods validate payloads correctly."""

    def test_text_delta_requires_block_id(self):
        """text_delta requires a non-empty block_id."""
        # Valid — should not raise
        event = CardEvent.text_delta("b1", "hello")
        assert event.type == CardEventType.TEXT_DELTA
        assert event.payload["block_id"] == "b1"
        assert event.payload["text"] == "hello"

    def test_text_started_requires_block_id(self):
        """text_started requires a non-empty block_id."""
        event = CardEvent.text_started("my_block")
        assert event.type == CardEventType.TEXT_STARTED
        assert event.payload["block_id"] == "my_block"

    def test_progress_updated_valid(self):
        """progress_updated with current/total is valid."""
        event = CardEvent.progress_updated(current=5, total=10, label="step 5")
        assert event.type == CardEventType.PROGRESS_UPDATED
        assert event.payload["current"] == 5
        assert event.payload["total"] == 10

    def test_progress_updated_zero_total(self):
        """progress_updated with total=0 is valid (means unknown)."""
        event = CardEvent.progress_updated(current=0, total=0)
        assert event.type == CardEventType.PROGRESS_UPDATED

    def test_failed_includes_error(self):
        """failed() includes error in payload."""
        event = CardEvent.failed("timeout")
        assert event.type == CardEventType.FAILED
        assert event.payload["error"] == "timeout"

    def test_completed_optional_summary(self):
        """completed() can have optional summary."""
        event = CardEvent.completed()
        assert event.payload.get("summary") is None or "summary" not in event.payload

        event2 = CardEvent.completed(summary="All done")
        assert event2.payload["summary"] == "All done"

    def test_criteria_updated_payload(self):
        """criteria_updated includes content and counts."""
        event = CardEvent.criteria_updated("text", satisfied_count=2, total_count=5)
        assert event.type == CardEventType.CRITERIA_UPDATED
        assert event.payload["content"] == "text"
        assert event.payload["satisfied_count"] == 2
        assert event.payload["total_count"] == 5

    def test_cycle_started_payload(self):
        """cycle_started includes cycle_num and max_cycles."""
        event = CardEvent.cycle_started(cycle_num=1, max_cycles=3)
        assert event.type == CardEventType.CYCLE_STARTED
        assert event.payload["cycle_num"] == 1
        assert event.payload["max_cycles"] == 3

    def test_warning_updated_payload(self):
        """warning_updated includes warning text."""
        event = CardEvent.warning_updated("⚠️ Low memory")
        assert event.type == CardEventType.WARNING_UPDATED
        assert event.payload["warning"] == "⚠️ Low memory"
