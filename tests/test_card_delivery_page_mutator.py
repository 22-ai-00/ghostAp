"""Unit tests for PageMutator: card create/update/stream/finalize operations."""

from dataclasses import dataclass
from unittest.mock import ANY, MagicMock, patch

from src.card.delivery.binding import BindingStore, PageBinding
from src.card.delivery.page_mutator import PageMutator
from src.card.delivery.sequence import SequenceManager
from src.card.delivery.types import SequenceConflictError, TransportError
from src.card.types import ActiveElement, RenderedCard


class MockClient:
    """Minimal CardAPIClient mock."""

    def __init__(self):
        self.creates = []
        self.updates = []
        self.elements = []
        self.finishes = []
        self.streaming_creates = []
        self.card_references = []
        self._counter = 0
        self._raise_on_create: Exception | None = None
        self._raise_on_update: Exception | None = None
        self._raise_on_element: Exception | None = None
        self._raise_on_streaming: Exception | None = None
        self._raise_on_finish: Exception | None = None

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        if self._raise_on_create:
            raise self._raise_on_create
        self._counter += 1
        return (f"msg_{self._counter}", f"card_{self._counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        if self._raise_on_update:
            raise self._raise_on_update
        self.updates.append(
            {"card_id": card_id, "card_json": card_json, "sequence": sequence}
        )

    def update_element(self, card_id, element_id, content, *, sequence=0):
        if self._raise_on_element:
            raise self._raise_on_element
        self.elements.append({"card_id": card_id, "element_id": element_id, "content": content})

    def create_streaming_card(self, card_json):
        if self._raise_on_streaming:
            raise self._raise_on_streaming
        self._counter += 1
        return f"stream_card_{self._counter}"

    def send_card_reference(self, chat_id, card_id, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._counter += 1
        return f"ref_msg_{self._counter}"

    def finish_streaming_card(self, card_id, *, sequence):
        if self._raise_on_finish:
            raise self._raise_on_finish
        self.finishes.append({"card_id": card_id, "sequence": sequence})


def _make_card(*, page_index=0, signature="sig_1", text="hello", streaming=False):
    card_json = {"body": {"elements": []}}
    if streaming:
        card_json["config"] = {"streaming_mode": True}
    active = ActiveElement(element_id="el_1", text=text) if text else None
    return RenderedCard(
        _card_json=card_json,
        structure_signature=signature,
        page_index=page_index,
        total_pages=1,
        active_element=active,
    )


def _make_streaming_payload(content: str) -> dict:
    return {
        "config": {"streaming_mode": True},
        "body": {
            "elements": [
                {"tag": "markdown", "element_id": "el_1", "content": content}
            ]
        },
    }


class TestCreatePage:
    def test_create_audit_fallback_preserves_logical_source_page(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)
        bindings.create("sess_1", "chat_1")
        calls = 0

        def reject_then_fallback(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TransportError(
                    "Card create failed: code=230028, msg=do not pass the audit",
                    code=230028,
                )
            return "fallback_msg", "fallback_card"

        client.create_card = reject_then_fallback
        outcome = mutator.create_page(
            "sess_1",
            "chat_1",
            _make_card(page_index=4),
            source_page_index=1,
        )

        assert outcome.kind == "applied"
        binding = bindings.get("sess_1")
        assert binding.pages[4].source_page_index == 1
        assert binding.latest_source_page_index == 1

    def test_create_page_success(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        # Pre-create binding (as CardDelivery.deliver() does before calling create_page)
        bindings.create("sess_1", "chat_1")

        card = _make_card()
        outcome = mutator.create_page("sess_1", "chat_1", card)

        assert outcome.kind == "applied"
        assert "created:" in outcome.message
        # Binding should be set
        binding = bindings.get("sess_1")
        assert binding is not None
        assert 0 in binding.pages
        assert binding.pages[0].is_streaming is False

    def test_create_page_streaming_success(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        card = _make_card(streaming=True)
        outcome = mutator.create_page("sess_1", "chat_1", card)

        assert outcome.kind == "applied"
        assert bindings.get("sess_1").pages[0].is_streaming is True

    def test_terminal_streaming_page_updates_snapshot_then_finishes(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)
        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old", "partial")
        page = bindings.get("sess_1").pages[0]
        page.is_streaming = True
        terminal = _make_card(signature="terminal", text="done", streaming=False)

        outcome = mutator.update_page(
            "sess_1",
            page,
            terminal,
            force_finish_streaming=True,
        )

        assert outcome.kind == "applied"
        assert client.updates[0]["card_json"]["config"]["streaming_mode"] is True
        assert client.updates[0]["sequence"] == 1
        assert client.finishes == [{"card_id": "card_1", "sequence": 2}]

    def test_finish_failure_keeps_stream_open_for_terminal_retry(self):
        client = MockClient()
        client._raise_on_finish = TransportError("finish unavailable")
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)
        bindings.create("sess_1", "chat_1")
        bindings.set_page(
            "sess_1",
            0,
            "msg_1",
            "card_1",
            "old",
            "partial",
            is_streaming=True,
        )
        page = bindings.get("sess_1").pages[0]

        outcome = mutator.update_page(
            "sess_1",
            page,
            _make_card(signature="terminal", text="done", streaming=True),
            force_finish_streaming=True,
        )

        assert outcome.kind == "reconcile"
        assert page.is_streaming is True
        assert page.signature == "old"

    def test_create_page_records_guarded_active_text(self):
        """Binding cache must match the text actually present in the sent card."""
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        card = RenderedCard(
            _card_json=_make_streaming_payload("full active text"),
            structure_signature="sig_1",
            page_index=0,
            total_pages=1,
            active_element=ActiveElement(element_id="el_1", text="full active text"),
        )

        with patch(
            "src.card.delivery.page_mutator._guard_payload",
            return_value=_make_streaming_payload("guarded active text"),
        ):
            outcome = mutator.create_page("sess_1", "chat_1", card)

        assert outcome.kind == "applied"
        binding = bindings.get("sess_1")
        assert binding.pages[0].last_text == "guarded active text"

    def test_create_page_redacts_email_like_content_before_send(self):
        """Feishu audit rejects EMAIL_ADDRESS; card delivery must redact before create."""
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        payload = {
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "element_id": "el_1",
                        "content": "contact admin@example.com for details",
                    }
                ]
            }
        }
        sent_payloads = []

        def create_card(chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
            sent_payloads.append(card_json)
            return ("msg_1", "card_1")

        client.create_card = create_card
        bindings.create("sess_1", "chat_1")
        card = RenderedCard(
            _card_json=payload,
            structure_signature="sig_email",
            page_index=0,
            total_pages=1,
            active_element=ActiveElement(element_id="el_1", text="contact admin@example.com for details"),
        )

        outcome = mutator.create_page("sess_1", "chat_1", card)

        assert outcome.kind == "applied"
        sent_content = sent_payloads[0]["body"]["elements"][0]["content"]
        assert "admin@example.com" not in sent_content
        assert "[redacted:email]" in sent_content
        binding = bindings.get("sess_1")
        assert binding is not None
        assert binding.pages[0].last_text == "contact [redacted:email] for details"

    def test_create_page_streaming_fallback_to_im(self):
        """When streaming creation fails, should fall back to IM create API."""
        client = MockClient()
        client._raise_on_streaming = RuntimeError("streaming failed")
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        card = _make_card(streaming=True)
        outcome = mutator.create_page("sess_1", "chat_1", card)

        assert outcome.kind == "applied"
        # Binding should still be set via fallback
        binding = bindings.get("sess_1")
        assert binding is not None
        assert 0 in binding.pages
        assert binding.pages[0].is_streaming is False

    def test_channel_streaming_failure_does_not_fallback_to_inline_card(self):
        client = MockClient()
        client._raise_on_streaming = RuntimeError("streaming failed")
        client.allow_streaming_fallback = False
        client.create_card = MagicMock(return_value=("msg_inline", "msg_inline"))
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)
        bindings.create("sess_1", "chat_1")

        outcome = mutator.create_page(
            "sess_1",
            "chat_1",
            _make_card(streaming=True),
        )

        assert outcome.kind == "reconcile"
        client.create_card.assert_not_called()
        assert bindings.get("sess_1").pages == {}

    def test_create_page_total_failure(self):
        """When all creation attempts fail, returns reconcile."""
        client = MockClient()
        client._raise_on_create = RuntimeError("API down")
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        card = _make_card()
        outcome = mutator.create_page("sess_1", "chat_1", card)

        assert outcome.kind == "reconcile"
        assert "API down" in outcome.message


class TestUpdatePage:
    def test_update_page_success(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        # Set up existing binding
        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(signature="new_sig")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "applied"
        assert "updated:" in outcome.message

    def test_update_page_records_guarded_active_text(self):
        """Full PATCH truncation must not poison last_text with unsent content."""
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]
        card = RenderedCard(
            _card_json=_make_streaming_payload("full active text"),
            structure_signature="new_sig",
            page_index=0,
            total_pages=1,
            active_element=ActiveElement(element_id="el_1", text="full active text"),
        )

        with patch(
            "src.card.delivery.page_mutator._guard_payload",
            return_value=_make_streaming_payload("guarded active text"),
        ):
            outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "applied"
        assert bindings.get("sess_1").pages[0].last_text == "guarded active text"

    def test_update_page_sequence_conflict_raises_floor(self):
        """On SequenceConflictError, should raise floor and return reconcile."""
        client = MockClient()
        client._raise_on_update = SequenceConflictError(next_floor=10)
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(signature="new_sig")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "reconcile"
        assert "sequence_conflict" in outcome.message

    def test_update_page_transport_error(self):
        """Transport errors return reconcile outcome."""
        client = MockClient()
        client._raise_on_update = TransportError("timeout")
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(signature="new_sig")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "reconcile"

    def test_update_page_content_invalid_patches_fallback_without_removing_binding(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        calls = []

        def update_card(card_id, card_json, *, sequence=0):
            calls.append({"card_id": card_id, "card_json": card_json, "sequence": sequence})
            if len(calls) == 1:
                raise TransportError(
                    "Patch failed: code=230099, msg=ErrCode: 200621; "
                    "ErrMsg: content parse failed",
                    code=230099,
                )

        client.update_card = update_card
        card = _make_card(signature="new_sig")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "applied"
        assert outcome.message == "fallback_content_invalid:230099"
        binding = bindings.get("sess_1")
        assert binding is not None
        assert 0 in binding.pages
        assert binding.pages[0].signature == "new_sig"
        assert len(calls) == 2
        assert calls[1]["card_json"]["header"]["title"]["content"] == "⚠️ 卡片渲染失败"
        assert calls[1]["card_json"]["body"]["elements"][0]["content"].endswith(
            "原因：code=230099；content parse failed"
        )

    def test_update_page_content_invalid_fallback_shows_table_limit_reason(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        calls = []

        def update_card(card_id, card_json, *, sequence=0):
            calls.append({"card_id": card_id, "card_json": card_json, "sequence": sequence})
            if len(calls) == 1:
                raise TransportError(
                    "Patch failed: code=230099, msg=Failed to create card content, "
                    "ext=ErrCode: 11310; ErrMsg: card table number over limit; "
                    "ErrorValue: table; ",
                    code=230099,
                )

        client.update_card = update_card
        card = _make_card(signature="new_sig")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "applied"
        assert "原因：code=230099；card table number over limit" in (
            calls[1]["card_json"]["body"]["elements"][0]["content"]
        )

    def test_update_page_content_invalid_fallback_failure_keeps_binding_state(
        self,
    ):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        calls = []

        def update_card(card_id, card_json, *, sequence=0):
            calls.append({"card_id": card_id, "card_json": card_json, "sequence": sequence})
            if len(calls) == 1:
                raise TransportError(
                    "Patch failed: code=230099, ErrCode: 11310; card table number over limit",
                    code=230099,
                )
            raise TransportError(
                "Patch failed: code=230099, ErrCode: 200830; schemaV2 card can not change schemaV1",
                code=230099,
            )

        client.update_card = update_card
        card = _make_card(signature="bad_sig")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "reconcile"
        assert outcome.message == "fallback_failed:230099"
        binding = bindings.get("sess_1")
        assert binding is not None
        assert binding.pages[0].signature == "old_sig"
        assert binding.pages[0].last_text == "old_text"
        assert len(calls) == 2

    def test_update_page_audit_fallback_failure_does_not_advance_binding(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page(
            "sess_1",
            0,
            "msg_1",
            "card_1",
            "old_sig",
            "old_text",
        )
        page = bindings.get("sess_1").pages[0]
        calls = []

        def update_card(card_id, card_json, *, sequence=0):
            calls.append(
                {
                    "card_id": card_id,
                    "card_json": card_json,
                    "sequence": sequence,
                }
            )
            if len(calls) == 1:
                raise TransportError(
                    "The messages do NOT pass the audit",
                    code=230028,
                )
            raise TransportError("fallback transport unavailable", code=1)

        client.update_card = update_card
        outcome = mutator.update_page(
            "sess_1",
            page,
            _make_card(signature="new_sig"),
        )

        assert outcome.kind == "reconcile"
        assert outcome.message == "fallback_audit_failed:1"
        binding = bindings.get("sess_1")
        assert binding is not None
        assert binding.pages[0].signature == "old_sig"
        assert binding.pages[0].last_text == "old_text"
        assert len(calls) == 2

    def test_update_page_sanitizes_only_markdown_image_references(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)
        bindings.create("sess_1", "chat_1")
        bindings.set_page(
            "sess_1",
            0,
            "msg_1",
            "card_1",
            "old_sig",
            "old_text",
        )
        page = bindings.get("sess_1").pages[0]
        long_alt = "evidence" * 100
        content = (
            f"before ![{long_alt}](/tmp/private/final.png) "
            "valid ![uploaded](img_v3_evidence)"
        )
        native_image = {
            "tag": "img",
            "img_key": "img_v3_native",
            "alt": {"tag": "plain_text", "content": "native"},
        }
        card = RenderedCard(
            _card_json={
                "body": {
                    "elements": [
                        {
                            "tag": "markdown",
                            "element_id": "el_1",
                            "content": content,
                        },
                        native_image,
                    ]
                }
            },
            structure_signature="new_sig",
            page_index=0,
            total_pages=1,
            active_element=ActiveElement(element_id="el_1", text=content),
        )

        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "applied"
        sent = client.updates[0]["card_json"]["body"]["elements"]
        safe_markdown = sent[0]["content"]
        assert "/tmp/private/final.png" not in safe_markdown
        assert "![uploaded](img_v3_evidence)" in safe_markdown
        assert len(safe_markdown) < 300
        assert sent[1] == native_image

    def test_stream_element_sanitizes_invalid_markdown_image_reference(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)
        bindings.create("sess_1", "chat_1")
        bindings.set_page(
            "sess_1",
            0,
            "msg_1",
            "card_1",
            "sig",
            "old_text",
        )
        page = bindings.get("sess_1").pages[0]
        text = (
            "done ![local](/tmp/private.png) "
            "![uploaded](img_v3_uploaded)"
        )

        outcome = mutator.stream_element(
            "sess_1",
            page,
            _make_card(text=text),
        )

        assert outcome.kind == "applied"
        sent = client.elements[0]["content"]
        assert "/tmp/private.png" not in sent
        assert "![uploaded](img_v3_uploaded)" in sent

    def test_invalid_image_key_message_is_content_invalid_without_code(self):
        error = TransportError(
            "card contains invalid image keys: /tmp/private.png",
            code=0,
        )

        assert error.is_content_invalid is True

    def test_update_page_audit_rejection_patches_safe_fallback(self):
        """Audit rejection is permanent for the raw payload; patch a safe terminal fallback."""
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        calls = []

        def update_card(card_id, card_json, *, sequence=0):
            calls.append({"card_id": card_id, "card_json": card_json, "sequence": sequence})
            if len(calls) == 1:
                raise TransportError(
                    "Patch failed: code=230028, msg=The messages do NOT pass the audit, "
                    "ext=contain sensitive data: EMAIL_ADDRESS",
                    code=230028,
                )

        client.update_card = update_card
        card = _make_card(signature="new_sig", text="hello admin@example.com")
        outcome = mutator.update_page("sess_1", page, card)

        assert outcome.kind == "applied"
        assert outcome.message == "fallback_audit_rejected:230028"
        assert len(calls) == 2
        fallback_payload = calls[1]["card_json"]
        assert fallback_payload["header"]["title"]["content"] == "⚠️ 卡片内容受限"
        fallback_content = fallback_payload["body"]["elements"][0]["content"]
        assert "EMAIL_ADDRESS" not in fallback_content
        assert "admin@example.com" not in fallback_content
        binding = bindings.get("sess_1")
        assert binding is not None
        assert binding.pages[0].signature == "new_sig"
        assert binding.pages[0].last_text == "card_audit_rejected"

    def test_update_page_missing_message_removes_binding_for_recreate(self):
        client = MockClient()
        client._raise_on_update = TransportError("message not found", code=99992354)
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        outcome = mutator.update_page("sess_1", page, _make_card(signature="new_sig"))

        assert outcome.kind == "reconcile"
        assert outcome.message == "recreate:99992354"
        assert 0 not in bindings.get("sess_1").pages


class TestStreamElement:
    def test_stream_element_success(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(text="new_text")
        outcome = mutator.stream_element("sess_1", page, card)

        assert outcome.kind == "applied"
        assert "element:" in outcome.message

    def test_stream_element_redacts_email_like_content_before_send(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(text="contact admin@example.com")
        outcome = mutator.stream_element("sess_1", page, card)

        assert outcome.kind == "applied"
        assert client.elements[0]["content"] == "contact [redacted:email]"
        assert bindings.get("sess_1").pages[0].last_text == "contact [redacted:email]"

    def test_stream_element_no_active_element(self):
        """Without active_element, returns skipped."""
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(text="")  # No active element
        # Need to set active_element to None explicitly
        card = RenderedCard(_card_json={"body": {}}, structure_signature="sig", page_index=0, total_pages=1, active_element=None)
        outcome = mutator.stream_element("sess_1", page, card)

        assert outcome.kind == "skipped"

    def test_stream_element_failure_falls_back_to_update(self):
        """When element update fails, should fall back to full update."""
        client = MockClient()
        client._raise_on_element = RuntimeError("element API broken")
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "old_text")
        page = bindings.get("sess_1").pages[0]

        card = _make_card(text="new_text", signature="new_sig")
        outcome = mutator.stream_element("sess_1", page, card)

        # Falls back to update_page, which should succeed
        assert outcome.kind == "applied"
        assert "updated:" in outcome.message


class TestFinalizePage:
    def test_finalize_page_resets_sequence_and_removes(self):
        client = MockClient()
        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        bindings.create("sess_1", "chat_1")
        bindings.set_page("sess_1", 0, "msg_1", "card_1", "sig", "text")
        page = bindings.get("sess_1").pages[0]

        mutator.finalize_page("sess_1", page)

        # Page should be removed from bindings
        binding = bindings.get("sess_1")
        assert 0 not in binding.pages


# ---------------------------------------------------------------------------
# Merged from test_page_mutator.py
# ---------------------------------------------------------------------------


@dataclass
class FakeActiveElement:
    element_id: str = "elm_1"
    text: str = "hello world"


@dataclass
class FakeRenderedCard:
    page_index: int = 0
    structure_signature: str = "sig_abc"
    active_element: FakeActiveElement | None = None

    def to_feishu_json(self):
        return {"config": {"streaming_mode": False}, "body": "fake"}


def _make_streaming_card():
    card = FakeRenderedCard(active_element=FakeActiveElement())
    card.to_feishu_json = lambda: {"config": {"streaming_mode": True}, "body": "stream"}
    return card


class TestCreatePageStreamingSuccess:
    """create_page: streaming path succeeds."""

    def test_streaming_card_creates_via_streaming_api(self):
        client = MagicMock()
        client.create_streaming_card.return_value = "card_id_1"
        client.send_card_reference.return_value = "msg_id_1"

        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        card = _make_streaming_card()
        result = mutator.create_page("session_1", "chat_1", card)

        assert result.kind == "applied"
        assert "msg_id_1" in result.message
        client.create_streaming_card.assert_called_once()
        client.send_card_reference.assert_called_once_with(
            "chat_1",
            "card_id_1",
            reply_to=None,
            reply_in_thread=None,
            idempotency_key=ANY,
        )


class TestCreatePageStreamingFallback:
    """create_page: streaming fails, falls back to IM API."""

    def test_streaming_failure_falls_back_to_create_card(self):
        client = MagicMock()
        client.create_streaming_card.side_effect = RuntimeError("streaming unavailable")
        client.create_card.return_value = ("msg_id_2", "card_id_2")

        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        card = _make_streaming_card()
        result = mutator.create_page("session_1", "chat_1", card)

        assert result.kind == "applied"
        assert "msg_id_2" in result.message
        client.create_card.assert_called_once()


class TestUpdatePageSequenceConflict:
    """update_page: SequenceConflictError triggers reconcile outcome."""

    def test_sequence_conflict_raises_floor_and_returns_reconcile(self):
        client = MagicMock()
        from src.card.delivery.engine import SequenceConflictError as SCE
        conflict = SCE(next_floor=10)
        client.update_card.side_effect = conflict

        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        page = PageBinding(page_index=0, message_id="msg_1", card_id="card_1", signature="old_sig", last_text="")
        card = FakeRenderedCard(active_element=FakeActiveElement())

        result = mutator.update_page("session_1", page, card)

        assert result.kind == "reconcile"
        assert "sequence_conflict" in result.message
        # Floor should have been raised
        assert sequences.next_sequence("card_1") >= 10


class TestStreamElementFallbackToUpdate:
    """stream_element: on failure, falls back to update_page."""

    def test_element_update_error_triggers_full_update_fallback(self):
        client = MagicMock()
        # First call (update_element) fails, second call (update_card in fallback) succeeds
        client.update_element.side_effect = RuntimeError("element API unavailable")
        client.update_card.return_value = None

        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        page = PageBinding(page_index=0, message_id="msg_1", card_id="card_1", signature="old_sig", last_text="")
        card = FakeRenderedCard(active_element=FakeActiveElement())

        result = mutator.stream_element("session_1", page, card)

        # Should fall back to update_page which succeeds
        assert result.kind == "applied"
        client.update_element.assert_called_once()
        client.update_card.assert_called_once()

    def test_sequence_conflict_in_element_falls_back_to_update(self):
        client = MagicMock()
        from src.card.delivery.engine import SequenceConflictError as SCE
        conflict = SCE(next_floor=5)
        client.update_element.side_effect = conflict
        client.update_card.return_value = None

        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        page = PageBinding(page_index=0, message_id="msg_1", card_id="card_1", signature="old_sig", last_text="")
        card = FakeRenderedCard(active_element=FakeActiveElement())

        result = mutator.stream_element("session_1", page, card)

        assert result.kind == "applied"
        client.update_card.assert_called_once()


class TestInvalidCardFallback:
    """update_page: invalid card content patches a known-good Schema V2 fallback."""

    def test_content_invalid_fallback_payload_keeps_schema_v2_identity(self):
        client = MagicMock()
        client.update_card.side_effect = [
            TransportError(
                "Patch failed: ErrCode: 200861; ErrPath: ROOT -> body -> elements -> [6](tag: note)",
                code=230099,
            ),
            None,
        ]

        bindings = BindingStore()
        sequences = SequenceManager()
        mutator = PageMutator(client, bindings, sequences)

        page = PageBinding(page_index=0, message_id="msg_1", card_id="card_1", signature="old_sig", last_text="")
        card = FakeRenderedCard(active_element=FakeActiveElement())

        result = mutator.update_page("session_1", page, card)

        assert result.kind == "applied"
        fallback_payload = client.update_card.call_args_list[1].args[1]
        assert fallback_payload["schema"] == "2.0"
        assert fallback_payload["config"]["update_multi"] is True
