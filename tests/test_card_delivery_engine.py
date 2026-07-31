"""Tests for CardDelivery engine."""

import dataclasses
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.card.delivery.engine import CardDelivery, SequenceConflictError, TransportError
from src.card.delivery.types import MutationOutcome as MutationOutcomeType
from src.card.programming_adapter import build_programming_metadata
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.state.models import CardState, ContentBlock, ImageBlock
from src.card.types import ActiveElement, RenderedCard
from tests.helpers.delivery_internals import DeliveryInspector


class MockCardClient:
    """Mock implementation of CardAPIClient."""

    def __init__(self):
        self.creates: list[dict] = []
        self.updates: list[dict] = []
        self.elements: list[dict] = []
        self.streaming_creates: list[dict] = []
        self.card_references: list[dict] = []
        self.finishes: list[dict] = []
        self._create_counter = 0
        self._raise_on_update: Exception | None = None
        self._raise_on_streaming_create: Exception | None = None

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._create_counter += 1
        msg_id = f"msg_{self._create_counter}"
        card_id = f"card_{self._create_counter}"
        self.creates.append({
            "chat_id": chat_id, "card_json": card_json,
            "reply_to": reply_to, "idempotency_key": idempotency_key,
        })
        return (msg_id, card_id)

    def update_card(self, card_id, card_json, *, sequence=0):
        if self._raise_on_update:
            raise self._raise_on_update
        self.updates.append({"card_id": card_id, "card_json": card_json, "sequence": sequence})

    def update_element(self, card_id, element_id, content, *, sequence=0):
        if self._raise_on_update:
            raise self._raise_on_update
        self.elements.append({"card_id": card_id, "element_id": element_id, "content": content, "sequence": sequence})

    def create_streaming_card(self, card_json):
        if self._raise_on_streaming_create:
            raise self._raise_on_streaming_create
        self._create_counter += 1
        card_id = f"stream_card_{self._create_counter}"
        self.streaming_creates.append({"card_json": card_json})
        return card_id

    def send_card_reference(self, chat_id, card_id, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._create_counter += 1
        msg_id = f"msg_{self._create_counter}"
        self.card_references.append({
            "chat_id": chat_id, "card_id": card_id,
            "reply_to": reply_to, "idempotency_key": idempotency_key,
        })
        return msg_id

    def finish_streaming_card(self, card_id, *, sequence):
        self.finishes.append({"card_id": card_id, "sequence": sequence})


class TestCardDeliveryCreate:
    """First delivery creates cards."""

    def test_first_deliver_creates_card_and_binding(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        rendered = [RenderedCard(
            _card_json={"body": {"elements": []}},
            structure_signature="sig_1", page_index=0, total_pages=1,
        )]

        outcomes = delivery.deliver("sess_1", "chat_abc", rendered)
        assert len(outcomes) == 1
        assert outcomes[0].kind == "applied"
        assert len(client.creates) == 1
        assert client.creates[0]["chat_id"] == "chat_abc"

        binding = delivery.get_binding("sess_1")
        assert binding is not None
        assert 0 in binding.pages
        assert binding.pages[0].signature == "sig_1"


class TestCardDeliveryUpdate:
    """Subsequent deliveries compare signatures."""

    @pytest.mark.parametrize(
        ("replacement_key", "replacement_alt"),
        [
            pytest.param("img_new", "旧图片", id="image-key"),
            pytest.param("img_old", "新图片", id="alt"),
        ],
    )
    def test_replacing_image_render_fields_triggers_card_update(
        self,
        replacement_key,
        replacement_alt,
    ):
        client = MockCardClient()
        delivery = CardDelivery(client)
        initial = CardState(
            blocks=(
                ImageBlock(
                    block_id="image:artifact",
                    image_key="img_old",
                    alt="旧图片",
                    status="completed",
                ),
            ),
        )
        replacement = CardState(
            blocks=(
                ImageBlock(
                    block_id="image:artifact",
                    image_key=replacement_key,
                    alt=replacement_alt,
                    status="completed",
                ),
            ),
        )

        initial_card = render_card(initial, RenderBudget())[0]
        replacement_card = render_card(replacement, RenderBudget())[0]
        delivery.deliver("image_session", "chat_abc", [initial_card])

        outcomes = delivery.deliver(
            "image_session",
            "chat_abc",
            [replacement_card],
        )

        assert replacement_card.structure_signature != initial_card.structure_signature
        assert outcomes[0].kind == "applied"
        assert client.updates[0]["card_json"]["body"]["elements"][0] == {
            "tag": "img",
            "img_key": replacement_key,
            "alt": {"tag": "plain_text", "content": replacement_alt},
        }

    def test_signature_change_triggers_update(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        r2 = [RenderedCard(_card_json={"updated": True}, structure_signature="sig_2", page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)

        assert outcomes[0].kind == "applied"
        assert len(client.updates) == 1

    def test_text_only_triggers_element_content(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        active = ActiveElement(element_id="el_1", text="hello")
        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", active_element=active, page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        active2 = ActiveElement(element_id="el_1", text="hello world")
        r2 = [RenderedCard(_card_json={}, structure_signature="sig_1", active_element=active2, page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)

        assert outcomes[0].kind == "applied"
        assert len(client.elements) == 1
        assert client.elements[0]["content"] == "hello world"

    def test_nested_programming_section_uses_element_content_update(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        metadata = build_programming_metadata("codex")
        initial_state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="stream",
                    content="正在检查",
                    element_id="el_stream",
                    status="active",
                ),
            ),
            metadata=metadata,
        )
        updated_state = dataclasses.replace(
            initial_state,
            blocks=(
                dataclasses.replace(
                    initial_state.blocks[0],
                    content="正在检查流式投递",
                ),
            ),
        )

        initial = render_card(initial_state, RenderBudget())
        updated = render_card(updated_state, RenderBudget())
        delivery.deliver("programming-section", "chat_abc", initial)
        outcomes = delivery.deliver(
            "programming-section",
            "chat_abc",
            updated,
        )

        assert initial[0].structure_signature == updated[0].structure_signature
        assert outcomes[0].kind == "applied"
        assert len(client.streaming_creates) == 1
        assert len(client.card_references) == 1
        assert client.updates == []
        assert client.elements == [
            {
                "card_id": "stream_card_1",
                "element_id": "el_stream",
                "content": "正在检查流式投递",
                "sequence": 1,
            },
        ]

    def test_live_elapsed_does_not_disable_programming_element_streaming(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        metadata = dataclasses.replace(
            build_programming_metadata("codex"),
            session_started_at=100.0,
        )
        initial_state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="stream",
                    content="正在检查",
                    element_id="el_stream",
                    status="active",
                ),
            ),
            metadata=metadata,
        )
        updated_state = dataclasses.replace(
            initial_state,
            blocks=(
                dataclasses.replace(
                    initial_state.blocks[0],
                    content="正在检查流式投递",
                ),
            ),
        )

        with patch("src.card.render.footer.time.monotonic", return_value=220.0):
            initial = render_card(initial_state, RenderBudget())
        with patch("src.card.render.footer.time.monotonic", return_value=221.0):
            updated = render_card(updated_state, RenderBudget())

        delivery.deliver("elapsed-programming-section", "chat_abc", initial)
        outcomes = delivery.deliver(
            "elapsed-programming-section",
            "chat_abc",
            updated,
        )

        assert initial[0].structure_signature == updated[0].structure_signature
        assert outcomes[0].kind == "applied"
        assert client.updates == []
        assert client.elements == [
            {
                "card_id": "stream_card_1",
                "element_id": "el_stream",
                "content": "正在检查流式投递",
                "sequence": 1,
            },
        ]

    def test_live_elapsed_mask_keeps_other_footer_metadata_in_signature(self):
        metadata = dataclasses.replace(
            build_programming_metadata("codex"),
            is_subagent=True,
            model_name="model-a",
            unit_id="task-1",
            unit_label="核对投递",
            session_started_at=100.0,
        )
        initial_state = CardState(
            blocks=(
                ContentBlock(
                    kind="text",
                    block_id="stream",
                    content="正在检查",
                    element_id="el_stream",
                    status="active",
                ),
            ),
            metadata=metadata,
        )
        updated_state = dataclasses.replace(
            initial_state,
            blocks=(
                dataclasses.replace(
                    initial_state.blocks[0],
                    content="正在检查流式投递",
                ),
            ),
            metadata=dataclasses.replace(metadata, model_name="model-b"),
        )

        with patch("src.card.render.footer.time.monotonic", return_value=220.0):
            initial = render_card(initial_state, RenderBudget())
        with patch("src.card.render.footer.time.monotonic", return_value=221.0):
            updated = render_card(updated_state, RenderBudget())

        assert initial[0].structure_signature != updated[0].structure_signature

    def test_no_change_skips(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        active = ActiveElement(element_id="el_1", text="same")
        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", active_element=active, page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        outcomes = delivery.deliver("sess_1", "chat_abc", r1)
        assert outcomes[0].kind == "skipped"
        assert len(client.updates) == 0
        assert len(client.elements) == 0


class TestSequenceConflict:
    def test_sequence_conflict_reconcile(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        client._raise_on_update = SequenceConflictError(next_floor=10)
        r2 = [RenderedCard(_card_json={}, structure_signature="sig_2", page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)

        assert outcomes[0].kind == "reconcile"
        assert "sequence_conflict" in outcomes[0].message


class TestMultiPage:
    def test_multi_page_delivery(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        rendered = [
            RenderedCard(_card_json={"page": 0}, structure_signature="sig_1", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1}, structure_signature="sig_1", page_index=1, total_pages=2),
        ]
        outcomes = delivery.deliver("sess_1", "chat_abc", rendered)
        assert len(outcomes) == 2
        assert all(o.kind == "applied" for o in outcomes)

    def test_first_five_page_delivery_creates_messages_in_page_order(self):
        """Initial pagination must never create tail pages before history pages."""
        client = MockCardClient()
        delivery = CardDelivery(client)

        rendered = [
            RenderedCard(
                _card_json={"page": page_index},
                structure_signature=f"sig_{page_index}",
                page_index=page_index,
                total_pages=5,
            )
            for page_index in range(5)
        ]

        outcomes = delivery.deliver("deep_five_pages", "chat_abc", rendered)

        assert [item["card_json"]["page"] for item in client.creates] == [0, 1, 2, 3, 4]
        assert [outcome.kind for outcome in outcomes] == ["applied"] * 5

    def test_initial_page_failure_stops_before_creating_later_messages(self):
        """A failed earlier page must fence all later Feishu message creation."""
        client = MockCardClient()
        delivery = CardDelivery(client)
        real_create = client.create_card
        attempts = 0

        def fail_first_create(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TransportError("first page unavailable")
            return real_create(*args, **kwargs)

        client.create_card = fail_first_create
        rendered = [
            RenderedCard(
                _card_json={"page": page_index},
                structure_signature=f"sig_{page_index}",
                page_index=page_index,
                total_pages=3,
            )
            for page_index in range(3)
        ]

        outcomes = delivery.deliver("deep_failed_first", "chat_abc", rendered)

        assert attempts == 1
        assert client.creates == []
        assert [outcome.kind for outcome in outcomes] == ["reconcile"]

    def test_new_page_created_on_growth(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        r2 = [
            RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=2),
            RenderedCard(_card_json={}, structure_signature="sig_1", page_index=1, total_pages=2),
        ]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)
        assert outcomes[1].kind == "applied"
        assert len(client.creates) == 2

    def test_growth_flushes_previous_latest_page_once_before_freezing_it(self):
        """The page-boundary chunk must reach the old latest message before append."""
        client = MockCardClient()
        delivery = CardDelivery(client)

        delivery.deliver(
            "deep_growth",
            "chat_abc",
            [RenderedCard(
                _card_json={"page": 0, "content": "before-boundary"},
                structure_signature="p0_before",
                page_index=0,
                total_pages=1,
            )],
        )

        outcomes = delivery.deliver(
            "deep_growth",
            "chat_abc",
            [
                RenderedCard(
                    _card_json={"page": 0, "content": "includes-boundary"},
                    structure_signature="p0_boundary",
                    page_index=0,
                    total_pages=2,
                ),
                RenderedCard(
                    _card_json={"page": 1, "content": "after-boundary"},
                    structure_signature="p1_live",
                    page_index=1,
                    total_pages=2,
                ),
            ],
        )

        assert [outcome.kind for outcome in outcomes] == ["applied", "applied"]
        assert client.updates[-1]["card_json"]["content"] == "includes-boundary"
        binding = delivery.get_binding("deep_growth")
        assert binding.pages[0].is_frozen is True
        assert binding.pages[1].is_frozen is False

    def test_existing_history_pages_are_not_updated_after_continuation(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        first = [
            RenderedCard(_card_json={"page": 0, "title": "old"}, structure_signature="p0_t1", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "title": "latest 1s"}, structure_signature="p1_t1", page_index=1, total_pages=2),
        ]
        delivery.deliver("sess_1", "chat_abc", first)

        second = [
            RenderedCard(_card_json={"page": 0, "title": "old 2s"}, structure_signature="p0_t2", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "title": "latest 2s"}, structure_signature="p1_t2", page_index=1, total_pages=2),
        ]
        outcomes = delivery.deliver("sess_1", "chat_abc", second)
        assert [o.kind for o in outcomes] == ["skipped", "applied"]

    def test_terminal_delivery_updates_only_latest_page(self):
        """Historical messages stay snapshots; only the newest card reaches terminal."""
        client = MockCardClient()
        delivery = CardDelivery(client)

        running = [
            RenderedCard(_card_json={"page": 0, "hdr": "running"}, structure_signature="p0_run", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "hdr": "running"}, structure_signature="p1_run", page_index=1, total_pages=2),
        ]
        delivery.deliver("deep_sess", "chat_abc", running)

        terminal = [
            RenderedCard(_card_json={"page": 0, "hdr": "completed"}, structure_signature="p0_done", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "hdr": "completed", "summary": "FINAL"}, structure_signature="p1_done", page_index=1, total_pages=2),
        ]
        outcomes = delivery.deliver("deep_sess", "chat_abc", terminal, is_terminal=True)

        assert [o.kind for o in outcomes] == ["skipped", "applied"]
        assert outcomes[0].message == "history_page_frozen"
        patched_pages = [u["card_json"].get("page") for u in client.updates]
        assert patched_pages == [1]

    def test_non_terminal_still_freezes_history_pages(self):
        """Streaming (non-terminal) updates keep freezing history pages to avoid
        churn — the terminal exemption must not regress this optimization."""
        client = MockCardClient()
        delivery = CardDelivery(client)

        first = [
            RenderedCard(_card_json={"page": 0, "title": "old"}, structure_signature="p0_t1", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "title": "latest 1s"}, structure_signature="p1_t1", page_index=1, total_pages=2),
        ]
        delivery.deliver("sess_1", "chat_abc", first)

        second = [
            RenderedCard(_card_json={"page": 0, "title": "old 2s"}, structure_signature="p0_t2", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "title": "latest 2s"}, structure_signature="p1_t2", page_index=1, total_pages=2),
        ]
        outcomes = delivery.deliver("sess_1", "chat_abc", second, is_terminal=False)
        assert [o.kind for o in outcomes] == ["skipped", "applied"]


class TestPageShrink:
    def test_shrink_keeps_message_high_watermark_and_updates_latest_message(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r2 = [
            RenderedCard(_card_json={"page": 0}, structure_signature="sig_1", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1}, structure_signature="sig_1", page_index=1, total_pages=2),
        ]
        delivery.deliver("sess_1", "chat_abc", r2)

        r1 = [RenderedCard(
            _card_json={"content": "final compact state"},
            structure_signature="sig_2",
            page_index=0,
            total_pages=1,
        )]
        outcomes = delivery.deliver("sess_1", "chat_abc", r1, is_terminal=True)

        binding = delivery.get_binding("sess_1")
        assert set(binding.pages) == {0, 1}
        assert outcomes[-1].kind == "applied"
        assert client.updates[-1]["card_id"] == "card_2"
        assert client.updates[-1]["card_json"]["content"] == "final compact state"

    def test_shrink_relabels_latest_message_with_visible_high_watermark_page(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        delivery.deliver("sess_1", "chat_abc", [
            RenderedCard(
                _card_json={"header": {"title": {"content": f"Deep 任务 · 页 {idx + 1}/3"}}},
                structure_signature=f"sig_{idx}",
                page_index=idx,
                total_pages=3,
            )
            for idx in range(3)
        ])

        compact = RenderedCard(
            _card_json={"header": {"title": {"content": "Deep 任务"}}},
            structure_signature="sig_compact",
            page_index=0,
            total_pages=1,
        )
        delivery.deliver("sess_1", "chat_abc", [compact], is_terminal=True)

        assert client.updates[-1]["card_id"] == "card_3"
        assert client.updates[-1]["card_json"]["header"]["title"]["content"] == (
            "Deep 任务 · 页 3/3"
        )

    def test_three_to_two_page_shrink_preserves_ordered_content_across_messages(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        delivery.deliver("sess_1", "chat_abc", [
            RenderedCard(
                _card_json={"content": marker},
                structure_signature=f"old_{marker}",
                page_index=idx,
                total_pages=3,
            )
            for idx, marker in enumerate(("A", "B", "C"))
        ])
        delivery.deliver("sess_1", "chat_abc", [
            RenderedCard(
                _card_json={"content": content},
                structure_signature=f"compact_{idx}",
                page_index=idx,
                total_pages=2,
            )
            for idx, content in enumerate(("AB", "CD"))
        ])

        remote_payloads = [
            client.creates[0]["card_json"],
            client.creates[1]["card_json"],
            client.updates[-1]["card_json"],
        ]
        visible_content = "|".join(payload["content"] for payload in remote_payloads)
        positions = [visible_content.index(marker) for marker in "ABCD"]

        assert positions == sorted(positions)
        assert client.updates[-1]["card_id"] == "card_3"


class TestStreamingFallback:
    def test_streaming_success_uses_cardkit_path(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        active = ActiveElement(element_id="el_1", text="streaming text")
        rendered = [RenderedCard(
            _card_json={"config": {"streaming_mode": True}, "body": {}},
            structure_signature="sig_1", active_element=active, page_index=0, total_pages=1,
        )]

        outcomes = delivery.deliver("sess_1", "chat_abc", rendered)
        assert outcomes[0].kind == "applied"
        assert len(client.streaming_creates) == 1
        assert len(client.card_references) == 1
        assert len(client.creates) == 0

    def test_streaming_fallback_to_im_api(self):
        client = MockCardClient()
        client._raise_on_streaming_create = RuntimeError("CardKit unavailable")
        delivery = CardDelivery(client)

        active = ActiveElement(element_id="el_1", text="fallback text")
        rendered = [RenderedCard(
            _card_json={"config": {"streaming_mode": True}, "body": {}},
            structure_signature="sig_1", active_element=active, page_index=0, total_pages=1,
        )]

        outcomes = delivery.deliver("sess_1", "chat_abc", rendered)
        assert outcomes[0].kind == "applied"
        assert len(client.creates) == 1

    def test_paginated_streams_finish_when_frozen_and_terminal(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        running = [
            RenderedCard(
                _card_json={"config": {"streaming_mode": True}, "body": {}},
                structure_signature=f"run_{index}",
                page_index=index,
                total_pages=2,
            )
            for index in range(2)
        ]

        delivery.deliver("sess_stream_pages", "chat_abc", running)

        assert len(client.finishes) == 1
        first_finished = client.finishes[0]["card_id"]
        binding = delivery.get_binding("sess_stream_pages")
        assert binding.pages[0].is_streaming is False
        assert binding.pages[1].is_streaming is True

        terminal = [
            RenderedCard(
                _card_json={"config": {"streaming_mode": True}, "body": {}},
                structure_signature=f"done_{index}",
                page_index=index,
                total_pages=2,
            )
            for index in range(2)
        ]
        delivery.deliver(
            "sess_stream_pages",
            "chat_abc",
            terminal,
            is_terminal=True,
        )

        assert len(client.finishes) == 2
        assert {item["card_id"] for item in client.finishes} == {
            first_finished,
            binding.pages[1].card_id,
        }


class TestClose:
    def test_close_removes_binding_and_is_idempotent(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)
        assert delivery.get_binding("sess_1") is not None

        delivery.close("sess_1")
        assert delivery.get_binding("sess_1") is None

        # Second close does not raise
        delivery.close("sess_1")
        assert delivery.get_binding("sess_1") is None

        # Deliver after close is no-op
        outcomes = delivery.deliver("sess_1", "chat_abc", r1)
        assert outcomes == []


class TestTransportError:
    def test_transport_error_returns_reconcile(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        client._raise_on_update = TransportError("connection timeout")
        r2 = [RenderedCard(_card_json={}, structure_signature="sig_2", page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)

        assert outcomes[0].kind == "reconcile"
        assert "connection timeout" in outcomes[0].message

    def test_stale_latest_binding_preserves_history_and_appends_replacement(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        rendered_v1 = [
            RenderedCard(_card_json={"page": 0}, structure_signature="sig_1", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1}, structure_signature="sig_1", page_index=1, total_pages=2),
        ]
        delivery.deliver("sess_stale", "chat_abc", rendered_v1)

        update_attempts = {"n": 0}

        def stale_update(*args, **kwargs):
            update_attempts["n"] += 1
            raise TransportError("message not found", code=99992354)

        client.update_card = stale_update
        rendered_v2 = [
            RenderedCard(_card_json={"page": 0, "v": 2}, structure_signature="sig_2", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1, "v": 2}, structure_signature="sig_2", page_index=1, total_pages=2),
        ]
        outcomes = delivery.deliver("sess_stale", "chat_abc", rendered_v2)

        assert [outcome.kind for outcome in outcomes] == ["skipped", "applied"]
        assert update_attempts["n"] == 1
        binding = delivery.get_binding("sess_stale")
        assert binding is not None
        assert set(binding.pages) == {0, 2}
        assert binding.pages[0].is_frozen is True
        assert binding.message_high_watermark == 2

        client.update_card = MockCardClient.update_card.__get__(client, MockCardClient)
        outcomes = delivery.deliver("sess_stale", "chat_abc", rendered_v2)
        assert [outcome.kind for outcome in outcomes] == ["skipped", "skipped"]
        assert [item["card_json"]["page"] for item in client.creates] == [0, 1, 1]

    def test_stale_compacted_latest_appends_replacement_without_history_waterfall(
        self, caplog
    ):
        """A stale compacted live page must not expose each history page to PATCH."""
        client = MockCardClient()
        delivery = CardDelivery(client)
        delivery.deliver(
            "sess_compacted_stale",
            "chat_abc",
            [
                RenderedCard(
                    _card_json={"page": page_index},
                    structure_signature=f"old_{page_index}",
                    page_index=page_index,
                    total_pages=3,
                )
                for page_index in range(3)
            ],
        )

        update_attempts = 0

        def stale_once(*args, **kwargs):
            nonlocal update_attempts
            update_attempts += 1
            if update_attempts == 1:
                raise TransportError("message not found", code=99992354)
            return MockCardClient.update_card(client, *args, **kwargs)

        client.update_card = stale_once
        compact = RenderedCard(
            _card_json={
                "header": {"title": {"content": "Deep 任务"}},
                "content": "final compact state",
            },
            structure_signature="final_compact",
            page_index=0,
            total_pages=1,
        )

        with caplog.at_level("WARNING"):
            outcomes = delivery.deliver(
                "sess_compacted_stale",
                "chat_abc",
                [compact],
                is_terminal=True,
            )

        assert [outcome.kind for outcome in outcomes] == ["applied"]
        assert update_attempts == 1
        assert len(client.creates) == 4
        assert client.creates[-1]["card_json"]["content"] == "final compact state"
        assert client.creates[-1]["card_json"]["header"]["title"]["content"].endswith(
            "页 4/4"
        )
        binding = delivery.get_binding("sess_compacted_stale")
        assert binding.message_high_watermark == 3
        assert set(binding.pages) == {0, 1, 3}
        assert not [record for record in caplog.records if record.levelname == "WARNING"]

        delivery.deliver(
            "sess_compacted_stale",
            "chat_abc",
            [dataclasses.replace(compact, structure_signature="final_compact_v2")],
            is_terminal=True,
        )
        assert update_attempts == 2
        assert client.updates[-1]["card_id"] == "card_4"

    def test_stale_during_growth_keeps_boundary_and_latest_in_distinct_messages(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        delivery.deliver(
            "sess_stale_growth",
            "chat_abc",
            [RenderedCard(
                _card_json={"content": "before boundary"},
                structure_signature="before_boundary",
                page_index=0,
                total_pages=1,
            )],
        )

        def stale_update(*args, **kwargs):
            raise TransportError("message not found", code=99992354)

        client.update_card = stale_update
        outcomes = delivery.deliver(
            "sess_stale_growth",
            "chat_abc",
            [
                RenderedCard(
                    _card_json={"content": "complete boundary"},
                    structure_signature="complete_boundary",
                    page_index=0,
                    total_pages=2,
                ),
                RenderedCard(
                    _card_json={"content": "new latest"},
                    structure_signature="new_latest",
                    page_index=1,
                    total_pages=2,
                ),
            ],
        )

        assert [outcome.kind for outcome in outcomes] == ["applied", "applied"]
        assert [item["card_json"]["content"] for item in client.creates] == [
            "before boundary",
            "complete boundary",
            "new latest",
        ]
        binding = delivery.get_binding("sess_stale_growth")
        assert binding.message_high_watermark == 2
        assert set(binding.pages) == {1, 2}
        assert binding.pages[1].is_frozen is True
        assert binding.pages[2].is_frozen is False

        client.update_card = MockCardClient.update_card.__get__(client, MockCardClient)
        follow_up = delivery.deliver(
            "sess_stale_growth",
            "chat_abc",
            [
                RenderedCard(
                    _card_json={"content": "old history"},
                    structure_signature="old_history",
                    page_index=0,
                    total_pages=3,
                ),
                RenderedCard(
                    _card_json={"content": "second boundary complete"},
                    structure_signature="second_boundary",
                    page_index=1,
                    total_pages=3,
                ),
                RenderedCard(
                    _card_json={"content": "newest latest"},
                    structure_signature="newest_latest",
                    page_index=2,
                    total_pages=3,
                ),
            ],
        )

        assert [outcome.kind for outcome in follow_up] == [
            "skipped",
            "applied",
            "applied",
        ]
        assert client.updates[-1]["card_id"] == "card_3"
        assert client.updates[-1]["card_json"]["content"] == (
            "second boundary complete"
        )
        assert client.creates[-1]["card_json"]["content"] == "newest latest"
        binding = delivery.get_binding("sess_stale_growth")
        assert binding.message_high_watermark == 3
        assert binding.pages[2].source_page_index == 1
        assert binding.pages[2].is_frozen is True
        assert binding.pages[3].source_page_index == 2
        assert binding.pages[3].is_frozen is False

    def test_failed_single_page_replacement_retries_reserved_visible_page(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        original_create = client.create_card
        delivery.deliver(
            "sess_stale_retry",
            "chat_abc",
            [RenderedCard(
                _card_json={"content": "initial"},
                structure_signature="initial",
                page_index=0,
                total_pages=1,
            )],
        )

        client.update_card = lambda *args, **kwargs: (_ for _ in ()).throw(
            TransportError("message not found", code=99992354)
        )
        attempted_keys: list[str | None] = []

        def fail_replacement(
            chat_id,
            card_json,
            *,
            reply_to=None,
            reply_in_thread=None,
            idempotency_key=None,
        ):
            attempted_keys.append(idempotency_key)
            raise TimeoutError("replacement create timed out")

        client.create_card = fail_replacement
        replacement = RenderedCard(
            _card_json={"content": "replacement"},
            structure_signature="replacement",
            page_index=0,
            total_pages=1,
        )
        first = delivery.deliver(
            "sess_stale_retry",
            "chat_abc",
            [replacement],
        )
        assert [outcome.kind for outcome in first] == ["reconcile"]
        binding = delivery.get_binding("sess_stale_retry")
        assert binding.pages == {}
        assert binding.message_high_watermark == 0

        def successful_retry(*args, **kwargs):
            attempted_keys.append(kwargs.get("idempotency_key"))
            return original_create(*args, **kwargs)

        client.create_card = successful_retry
        second = delivery.deliver(
            "sess_stale_retry",
            "chat_abc",
            [replacement],
        )

        assert [outcome.kind for outcome in second] == ["applied"]
        assert attempted_keys[0] == attempted_keys[1]
        assert attempted_keys[0] != client.creates[0]["idempotency_key"]
        binding = delivery.get_binding("sess_stale_retry")
        assert binding.message_high_watermark == 1
        assert set(binding.pages) == {1}

    def test_transport_error_on_element_falls_back_to_update(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        active = ActiveElement(element_id="el_1", text="hello")
        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", active_element=active, page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        def failing_element(*args, **kwargs):
            raise TransportError("element push failed")

        client.update_element = failing_element
        active2 = ActiveElement(element_id="el_1", text="world")
        r2 = [RenderedCard(_card_json={}, structure_signature="sig_1", active_element=active2, page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)

        assert outcomes[0].kind == "applied"
        assert len(client.updates) == 1


class TestCreateCardFailure:
    def test_create_card_failure_returns_reconcile_no_binding(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        def failing_create(*args, **kwargs):
            raise RuntimeError("API unavailable")
        client.create_card = failing_create

        rendered = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", rendered)

        assert len(outcomes) == 1
        assert outcomes[0].kind == "reconcile"

        binding = delivery.get_binding("sess_1")
        if binding is not None:
            assert 0 not in binding.pages


class TestConcurrentDeliverClose:
    """Multi-threaded deliver/close race condition safety test."""

    def test_concurrent_deliver_and_close_no_exception(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        session_id = "concurrent_sess"
        errors: list[Exception] = []

        def deliver_task():
            try:
                cards = [RenderedCard(
                    page_index=0, _card_json={"header": {"title": {"content": "test"}}},
                    structure_signature="sig-concurrent",
                )]
                delivery.deliver(session_id, "chat_1", cards)
            except Exception as e:
                errors.append(e)

        def close_task():
            try:
                delivery.close(session_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=deliver_task if i % 2 == 0 else close_task)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        assert delivery.get_binding(session_id) is None


class TestCardDeliveryTOCTOU:
    def test_deliver_during_close_no_api_call_after_cleanup(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        session_id = "toctou_1"

        rendered = [RenderedCard(_card_json={"body": {}}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver(session_id, "chat_1", rendered)

        barrier = threading.Barrier(2, timeout=10)
        results: dict[str, list] = {"deliver": [], "errors": []}

        def deliver_thread():
            try:
                barrier.wait()
                time.sleep(0.02)
                outcome = delivery.deliver(session_id, "chat_1", rendered)
                results["deliver"] = outcome
            except Exception as e:
                results["errors"].append(e)

        def close_thread():
            try:
                barrier.wait()
                delivery.close(session_id)
            except Exception as e:
                results["errors"].append(e)

        t1 = threading.Thread(target=deliver_thread)
        t2 = threading.Thread(target=close_thread)
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

        assert not results["errors"]
        assert results["deliver"] == []
        assert len(client.creates) == 1


class TestDeliveryTimeout:
    def test_deliver_timeout_on_slow_api(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        def slow_create(*args, **kwargs):
            raise TimeoutError("Connection timed out after 30s")
        client.create_card = slow_create

        rendered = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", rendered)

        assert outcomes[0].kind == "reconcile"

    def test_update_timeout_returns_reconcile(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        r1 = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        delivery.deliver("sess_1", "chat_abc", r1)

        def timeout_update(*args, **kwargs):
            raise TimeoutError("Request timed out")
        client.update_card = timeout_update

        r2 = [RenderedCard(_card_json={}, structure_signature="sig_2", page_index=0, total_pages=1)]
        outcomes = delivery.deliver("sess_1", "chat_abc", r2)
        assert outcomes[0].kind == "reconcile"


class TestPartialMultipageFailure:
    def test_failed_next_page_keeps_boundary_live_for_terminal_compaction(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        original_create = client.create_card
        attempts = 0

        def fail_second_create(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise TimeoutError("page 1 create failed")
            return original_create(*args, **kwargs)

        client.create_card = fail_second_create
        first = delivery.deliver(
            "sess_partial_compaction",
            "chat_abc",
            [
                RenderedCard(
                    _card_json={"content": "boundary"},
                    structure_signature="boundary",
                    page_index=0,
                    total_pages=2,
                ),
                RenderedCard(
                    _card_json={"content": "latest"},
                    structure_signature="latest",
                    page_index=1,
                    total_pages=2,
                ),
            ],
        )

        assert [outcome.kind for outcome in first] == ["applied", "reconcile"]
        binding = delivery.get_binding("sess_partial_compaction")
        assert binding.pages[0].is_frozen is False

        client.create_card = original_create
        terminal = delivery.deliver(
            "sess_partial_compaction",
            "chat_abc",
            [RenderedCard(
                _card_json={"content": "final compact state"},
                structure_signature="final_compact",
                page_index=0,
                total_pages=1,
            )],
            is_terminal=True,
        )

        assert [outcome.kind for outcome in terminal] == ["applied"]
        assert client.updates[-1]["card_id"] == "card_1"
        assert client.updates[-1]["card_json"]["content"] == "final compact state"

    def test_partial_multipage_create_failure_and_retry(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        call_count = 0
        original_create = client.create_card

        def _failing_second_create(chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated page 1 creation failure")
            return original_create(chat_id, card_json, reply_to=reply_to, idempotency_key=idempotency_key)

        client.create_card = _failing_second_create

        rendered = [
            RenderedCard(_card_json={"page": 0}, structure_signature="sig_a", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1}, structure_signature="sig_b", page_index=1, total_pages=2),
        ]

        outcomes = delivery.deliver("sess_partial", "chat_abc", rendered)
        assert outcomes[0].kind == "applied"
        assert outcomes[1].kind == "reconcile"

        # Retry: fix client and deliver again
        client.create_card = original_create
        outcomes2 = delivery.deliver("sess_partial", "chat_abc", rendered)
        assert outcomes2[0].kind == "skipped"
        assert outcomes2[1].kind == "applied"

    def test_retry_missing_page_reuses_same_idempotency_key(self):
        client = MockCardClient()
        delivery = CardDelivery(client)

        call_count = 0
        original_create = client.create_card

        def _failing_second_create(chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                _failing_second_create.last_key = idempotency_key
                raise TimeoutError("lost")
            return original_create(chat_id, card_json, reply_to=reply_to, idempotency_key=idempotency_key)
        _failing_second_create.last_key = None

        client.create_card = _failing_second_create
        rendered = [
            RenderedCard(_card_json={"page": 0}, structure_signature="sig_a", page_index=0, total_pages=2),
            RenderedCard(_card_json={"page": 1}, structure_signature="sig_b", page_index=1, total_pages=2),
        ]

        delivery.deliver("sess_retry", "chat_abc", rendered)
        failed_key = _failing_second_create.last_key

        client.create_card = original_create
        delivery.deliver("sess_retry", "chat_abc", rendered)
        assert failed_key
        assert client.creates[-1]["idempotency_key"] == failed_key


class TestSessionLockEviction:
    def _make_delivery(self, *, max_locks: int = 10, lock_ttl: float = 600.0):
        client = MockCardClient()
        delivery = CardDelivery(client, max_session_locks=max_locks, session_lock_ttl=lock_ttl)
        return client, delivery

    def test_session_lock_evicted_after_close_and_ttl(self):
        import time as _time
        client, delivery = self._make_delivery(max_locks=10, lock_ttl=600.0)

        for i in range(9):
            sid = f"zombie_{i}"
            rendered = [RenderedCard(_card_json={}, structure_signature=f"sig_{i}", page_index=0, total_pages=1)]
            delivery.deliver(sid, "chat_1", rendered)

        inspector = DeliveryInspector.from_delivery(delivery)
        for i in range(9):
            inspector.timestamps[f"zombie_{i}"] = _time.monotonic() - 700.0
        for i in range(9):
            delivery._bindings.remove(f"zombie_{i}")

        assert len(inspector.session_locks) == 9
        inspector.evict_stale_two_phase()
        assert len(inspector.session_locks) < 9

    def test_session_lock_not_evicted_while_active(self):
        import time as _time
        client, delivery = self._make_delivery(max_locks=10, lock_ttl=600.0)

        for i in range(9):
            sid = f"active_{i}"
            rendered = [RenderedCard(_card_json={}, structure_signature=f"sig_{i}", page_index=0, total_pages=1)]
            delivery.deliver(sid, "chat_1", rendered)

        inspector = DeliveryInspector.from_delivery(delivery)
        for i in range(9):
            inspector.timestamps[f"active_{i}"] = _time.monotonic() - 700.0

        inspector.evict_stale_two_phase()
        for i in range(9):
            assert f"active_{i}" in inspector.session_locks


class TestEvictionConcurrencyStress:
    def test_concurrent_deliver_and_eviction_no_deadlock(self):
        client = MockCardClient()
        delivery = CardDelivery(client, max_session_locks=50, session_lock_ttl=1.0)
        delivery._eviction_interval = 0.05

        num_threads = 25
        iterations_per_thread = 20
        barrier = threading.Barrier(num_threads, timeout=30)
        errors: list[Exception] = []

        def worker(thread_id: int):
            try:
                barrier.wait()
                for i in range(iterations_per_thread):
                    session_id = f"stress_t{thread_id}_i{i}"
                    rendered = [RenderedCard(
                        _card_json={"body": {"elements": [{"tag": "markdown", "content": f"t{thread_id}i{i}"}]}},
                        structure_signature=f"sig_{thread_id}_{i}", page_index=0,
                    )]
                    try:
                        delivery.deliver(session_id, f"chat_{thread_id}", rendered)
                    except Exception:
                        pass
                    if i % 3 == 0:
                        try:
                            delivery.close(session_id)
                        except Exception:
                            pass
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        alive = [t for t in threads if t.is_alive()]
        assert not alive, f"{len(alive)} threads still alive"
        assert not errors
        delivery._shutdown()


class TestDeliverRejectedAtCapacity:
    def test_noop_degradation_when_capacity_full(self):
        client = MockCardClient()
        max_locks = 10
        delivery = CardDelivery(client, max_session_locks=max_locks)

        rendered = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        for i in range(max_locks):
            delivery.deliver(f"session_{i}", "chat_1", rendered)

        delivery._lock_pool._lru_evict_oldest = lambda: None
        result = delivery.deliver("session_overflow", "chat_1", rendered)
        assert result[0].kind == "applied"
        delivery._shutdown()

    def test_existing_session_not_rejected(self):
        client = MockCardClient()
        max_locks = 5
        delivery = CardDelivery(client, max_session_locks=max_locks)

        rendered = [RenderedCard(_card_json={}, structure_signature="sig_1", page_index=0, total_pages=1)]
        for i in range(max_locks):
            delivery.deliver(f"session_{i}", "chat_1", rendered)

        rendered_update = [RenderedCard(_card_json={"new": True}, structure_signature="sig_2", page_index=0, total_pages=1)]
        result = delivery.deliver("session_0", "chat_1", rendered_update)
        assert result[0].kind in ("applied", "reconcile")
        delivery._shutdown()


class TestDeliverEmptyRenderedOpenSession:
    def test_empty_rendered_new_session(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        try:
            outcomes = delivery.deliver("sess_empty", "chat_1", [])
            assert outcomes == []
            assert len(client.creates) == 0
        finally:
            delivery._shutdown()


class TestNonContiguousPageIndex:
    def test_gap_in_page_index_no_crash(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        try:
            rendered = [
                RenderedCard(_card_json={"body": {"elements": []}}, structure_signature="sig_0", page_index=0, total_pages=3),
                RenderedCard(_card_json={"body": {"elements": []}}, structure_signature="sig_2", page_index=2, total_pages=3),
            ]
            outcomes = delivery.deliver("sess_gap", "chat_1", rendered)
            assert len(outcomes) == 2
            assert len(client.creates) == 2
        finally:
            delivery._shutdown()


class TestTOCTOUClosedAfterLock:
    def test_closed_between_check_and_lock(self):
        client = MockCardClient()
        delivery = CardDelivery(client)
        try:
            rendered = [RenderedCard(
                _card_json={"body": {"elements": []}}, structure_signature="sig1", page_index=0, total_pages=1,
            )]
            delivery.deliver("toctou_sess", "chat_1", rendered)
            delivery.close("toctou_sess")
            outcomes = delivery.deliver("toctou_sess", "chat_1", rendered)
            assert outcomes == []
            assert len(client.creates) == 1
        finally:
            delivery._shutdown()


class TestSessionLockPoolFullScan:
    def test_full_scan_cleans_stale_locks(self):
        from src.card.delivery.lock_pool import SessionLockPool

        pool = SessionLockPool(max_locks=100, lock_ttl=0.1, eviction_interval=9999)
        try:
            for i in range(10):
                pool.acquire(f"s{i}")
            time.sleep(0.2)
            pool._last_full_scan = 0
            pool._full_scan_if_needed()
            assert pool.count == 0
        finally:
            pool.shutdown()

    def test_full_scan_skips_active_bindings(self):
        from src.card.delivery.lock_pool import SessionLockPool

        pool = SessionLockPool(
            max_locks=100, lock_ttl=0.1, eviction_interval=9999,
            has_active_binding=lambda sid: sid == "s_active",
        )
        try:
            pool.acquire("s_active")
            pool.acquire("s_stale")
            time.sleep(0.2)
            pool._last_full_scan = 0
            pool._full_scan_if_needed()
            assert pool.contains("s_active")
            assert not pool.contains("s_stale")
        finally:
            pool.shutdown()


class TestAsyncDeliveryExactlyOnce:
    def test_terminal_hook_fires_exactly_once_async(self):
        from src.card.delivery.engine import CardDelivery
        from src.card.events import CardEvent, CardEventType
        from src.card.session import CardSession
        from src.card.session.config import SessionCallbacks, SessionConfig
        from src.card.state.models import CardMetadata

        hook_calls = []
        hook_done = threading.Event()

        class _TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                hook_calls.append(reason)
                hook_done.set()

        client = MockCardClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", tool_name="test", model_name="m1")
        callbacks = SessionCallbacks(hooks=(_TrackingHook(),))
        config = SessionConfig(metadata=metadata, sync_delivery=False)

        session = CardSession(
            chat_id="async_test_chat", config=config,
            delivery=delivery, callbacks=callbacks, session_id="async_sess_1",
        )
        session._sync_delivery = False

        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent.completed(summary="done"))

        assert hook_done.wait(timeout=5.0)
        assert len(hook_calls) == 1
        assert session.closed


# ---------------------------------------------------------------------------
# Delivery close race (merged)
# ---------------------------------------------------------------------------


class _SlowCreateClient:
    def __init__(self, delay: float = 0.1):
        self._delay = delay
        self._create_counter = 0
        self.creates: list[dict] = []
        self.create_started = threading.Event()

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self.create_started.set()
        time.sleep(self._delay)
        self._create_counter += 1
        self.creates.append({"chat_id": chat_id})
        return (f"msg_{self._create_counter}", f"card_{self._create_counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        pass

    def update_element(self, card_id, element_id, content, *, sequence=0):
        pass

    def create_streaming_card(self, card_json):
        raise NotImplementedError

    def send_card_reference(self, chat_id, card_id, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        raise NotImplementedError


class TestDeliveryCloseRace:
    def _rendered(self, sig: str = "sig_1") -> list[RenderedCard]:
        return [RenderedCard(_card_json={"body": {"elements": []}}, structure_signature=sig, page_index=0, total_pages=1)]

    def test_close_during_create_card_no_exception(self):
        client = _SlowCreateClient(delay=0.15)
        delivery = CardDelivery(client)
        session_id = "race_1"
        errors: list[Exception] = []

        def deliver_thread():
            try:
                delivery.deliver(session_id, "chat_1", self._rendered())
            except Exception as e:
                errors.append(e)

        def close_thread():
            try:
                client.create_started.wait(timeout=5)
                delivery.close(session_id)
            except Exception as e:
                errors.append(e)

        t_deliver = threading.Thread(target=deliver_thread)
        t_close = threading.Thread(target=close_thread)
        t_deliver.start(); t_close.start()
        t_deliver.join(timeout=5); t_close.join(timeout=5)

        assert errors == []
        assert delivery.get_binding(session_id) is None
        delivery._shutdown()

    def test_per_session_lock_serializes_deliver_and_close(self):
        client = _SlowCreateClient(delay=0.2)
        delivery = CardDelivery(client)
        session_id = "race_5"
        deliver_outcomes: list = []
        close_done = threading.Event()

        def deliver_thread():
            deliver_outcomes.extend(delivery.deliver(session_id, "chat_1", self._rendered()))

        def close_thread():
            client.create_started.wait(timeout=5)
            delivery.close(session_id)
            close_done.set()

        t1 = threading.Thread(target=deliver_thread)
        t2 = threading.Thread(target=close_thread)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)

        assert len(deliver_outcomes) == 1
        assert deliver_outcomes[0].kind == "applied"
        assert close_done.is_set()
        assert delivery.get_binding(session_id) is None
        delivery._shutdown()


class TestCardDeliveryDestructor:
    def test_del_calls_lock_pool_shutdown(self):
        client = MagicMock()
        client.create_card = MagicMock(return_value=("msg_1", "card_1"))
        delivery = CardDelivery(client)
        mock_shutdown = MagicMock()
        delivery._lock_pool.shutdown = mock_shutdown
        delivery.__del__()
        mock_shutdown.assert_called_once()
        delivery._shutdown()

    def test_del_suppresses_exceptions(self):
        client = MagicMock()
        client.create_card = MagicMock(return_value=("msg_1", "card_1"))
        delivery = CardDelivery(client)
        delivery._lock_pool.shutdown = MagicMock(side_effect=RuntimeError("already shut down"))
        delivery.__del__()  # Should not raise
        delivery._lock_pool.shutdown = MagicMock()
        delivery._shutdown()


class TestMutationOutcomeStructure:
    def test_is_frozen_dataclass(self):
        outcome = MutationOutcomeType(kind="applied")
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.kind = "rejected"  # type: ignore[misc]

    @pytest.mark.parametrize("kind", ["applied", "reconcile", "skipped", "rejected"])
    def test_all_kinds_constructible(self, kind):
        o = MutationOutcomeType(kind=kind)
        assert o.kind == kind

    def test_default_message_is_empty(self):
        o = MutationOutcomeType(kind="applied")
        assert o.message == ""
