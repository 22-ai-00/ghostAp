from src.card.events import CardEvent
from src.card.state.models import CardMetadata, CardState, FooterState
from src.card.state.reducer import reduce_card_state


def test_card_split_event_carries_optional_bridge_phrase():
    event = CardEvent.card_split("task_done", bridge_phrase="续接：")

    assert event.payload["reason"] == "task_done"
    assert event.payload["bridge_phrase"] == "续接："


def test_continuation_seq_derives_visible_card_sequence():
    metadata = CardMetadata(continuation_seq=2)

    assert metadata.card_sequence == 3


def test_archived_reducer_freezes_metadata_with_pointer():
    state = CardState(
        metadata=CardMetadata(tool_name="coco", session_started_at=10.0),
        footer=FooterState(progress_started_at=20.0),
    )

    archived = reduce_card_state(
        state,
        CardEvent.archived(sequence=1, bridge_phrase="续接 #2 ↓"),
        state.metadata,
    )

    assert archived.metadata.frozen is True
    assert archived.metadata.final_state_for_freeze is state
    assert archived.footer.status_text == "本卡已停止更新 · 续接 #2 ↓"
