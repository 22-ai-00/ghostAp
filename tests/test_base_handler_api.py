"""Tests for BaseHandler unified messaging API.

Covers: reply_text, reply_card, update_card, send_card_to_chat, send_text_to_chat,
deprecated method stubs, _inject_ref_note post format, input validation,
and im_client default retry behavior.
"""
from __future__ import annotations

import base64
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handler_context import HandlerContext
from src.feishu.handlers.base import BaseHandler
from src.project.mapper import MessageLinker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(**settings_overrides) -> BaseHandler:
    """Create a BaseHandler with fully mocked context."""
    settings = MagicMock()
    settings.default_reply_mode = "direct"
    for k, v in settings_overrides.items():
        setattr(settings, k, v)

    ctx = HandlerContext(
        settings=settings,
        api_client_factory=MagicMock(),
        message_callback=MagicMock(),
        coco_manager=MagicMock(),
        claude_manager=MagicMock(),
        aiden_manager=MagicMock(),
        codex_manager=MagicMock(),
        gemini_manager=MagicMock(),
        traex_manager=MagicMock(),
        tui2acp_manager=MagicMock(),
        intent_recognizer=MagicMock(),
        scheduler=MagicMock(),
        project_manager=MagicMock(),
        message_mapper=MagicMock(),
        message_linker=MagicMock(),
        mode_manager=MagicMock(),
        context_manager=MagicMock(),
        deep_engine_manager=MagicMock(),
        progress_reporter=MagicMock(),
        spec_engine_manager=MagicMock(),
        spec_reporter=MagicMock(),
        thread_manager=MagicMock(),
        image_handler_factory=MagicMock(),
        working_dirs={},
        working_dir_lock=threading.Lock(),
        pending_image_keys={},
        pending_image_lock=threading.Lock(),
        enable_streaming=False,
        managers={},
        handlers={},
        slock_engine_manager=MagicMock(),
    )
    h = BaseHandler(ctx)
    # Mock the im_client (attached by handler_context or externally)
    h.im_client = MagicMock()
    return h


def _success_response(message_id: str = "resp_msg_001"):
    """Create a mock successful API response."""
    resp = MagicMock()
    resp.success.return_value = True
    resp.data.message_id = message_id
    return resp


def _failure_response():
    """Create a mock failed API response."""
    resp = MagicMock()
    resp.success.return_value = False
    return resp


def test_upload_acp_image_decodes_bounded_payload_for_im_client():
    from src.acp.models import ACPImageInfo

    handler = _make_handler()
    handler.im_client.upload_image_bytes.return_value = "img_generated"
    payload = b"\x89PNG\r\n\x1a\npayload"
    image = ACPImageInfo(
        image_id="sha256:image",
        mime_type="image/png",
        data=base64.b64encode(payload).decode("ascii"),
        name="generated.png",
    )

    image_key = handler.upload_acp_image(image)

    assert image_key == "img_generated"
    handler.im_client.upload_image_bytes.assert_called_once_with(payload)


def test_upload_acp_image_rejects_unsupported_mime_type():
    from src.acp.models import ACPImageInfo

    handler = _make_handler()
    image = ACPImageInfo(
        image_id="sha256:abc",
        mime_type="image/svg+xml",
        data=base64.b64encode(b"<svg/>").decode(),
        name="unsafe.svg",
    )

    assert handler.upload_acp_image(image) is None
    handler.im_client.upload_image_bytes.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "mime_type"),
    [
        (b"not an image", "image/png"),
        (b"\x89PNG\r\n\x1a\npayload", "image/jpeg"),
    ],
)
def test_upload_acp_image_defensively_rejects_spoofed_bytes(
    payload,
    mime_type,
):
    from src.acp.models import ACPImageInfo

    handler = _make_handler()
    image = ACPImageInfo(
        image_id="sha256:spoofed",
        mime_type=mime_type,
        data=base64.b64encode(payload).decode(),
        name="spoofed",
    )

    assert handler.upload_acp_image(image) is None
    handler.im_client.upload_image_bytes.assert_not_called()


def test_upload_acp_image_rejects_oversized_encoding_before_decode():
    from src.acp.client import MAX_ACP_IMAGE_BYTES
    from src.acp.models import ACPImageInfo

    handler = _make_handler()
    image = ACPImageInfo(
        image_id="sha256:oversized",
        mime_type="image/png",
        data="A" * ((((MAX_ACP_IMAGE_BYTES + 2) // 3) * 4) + 1),
        name="oversized.png",
    )

    with patch("src.feishu.handlers.base.base64.b64decode") as decode:
        assert handler.upload_acp_image(image) is None

    decode.assert_not_called()
    handler.im_client.upload_image_bytes.assert_not_called()


# ===========================================================================
# reply_text tests
# ===========================================================================


class TestReplyText:
    def test_success_returns_message_id(self):
        h = _make_handler()
        h.im_client.reply_message.return_value = _success_response("msg_123")
        result = h.reply_text("origin_msg", "hello")
        assert result == "msg_123"
        h.im_client.reply_message.assert_called_once()

    def test_passes_optional_idempotency_key_to_im_client(self):
        h = _make_handler()
        h.im_client.reply_message.return_value = _success_response("msg_123")

        result = h.reply_text(
            "origin_msg",
            "hello",
            idempotency_key="stable-reply-uuid",
        )

        assert result == "msg_123"
        assert h.im_client.reply_message.call_args.kwargs["idempotency_key"] == (
            "stable-reply-uuid"
        )

    def test_passes_canonical_dm_recipient_aliases_for_sibling_reply(self):
        h = _make_handler()
        linker = MessageLinker()
        linker.register_origin(
            "om_sibling",
            chat_id="oc_requester_dm",
            chat_type="p2p",
            sender_id="ou_requester",
        )
        h.ctx.message_linker = linker
        h.im_client.reply_message.return_value = _success_response("msg_123")

        result = h.reply_text("om_sibling", "hello")

        assert result == "msg_123"
        assert h.im_client.reply_message.call_args.kwargs["audit_aliases"] == (
            "oc_requester_dm",
            "ou_requester",
        )

    def test_none_input_returns_none(self, caplog):
        h = _make_handler()
        result = h.reply_text("origin_msg", None)
        assert result is None
        h.im_client.reply_message.assert_not_called()
        assert "空内容" in caplog.text

    def test_empty_string_input_returns_none(self, caplog):
        h = _make_handler()
        result = h.reply_text("origin_msg", "")
        assert result is None
        h.im_client.reply_message.assert_not_called()

    def test_api_exception_returns_none(self):
        h = _make_handler()
        h.im_client.reply_message.side_effect = RuntimeError("network error")
        result = h.reply_text("origin_msg", "hello")
        assert result is None

    def test_response_failure_returns_none(self):
        h = _make_handler()
        h.im_client.reply_message.return_value = _failure_response()
        result = h.reply_text("origin_msg", "hello")
        assert result is None


# ===========================================================================
# reply_card tests
# ===========================================================================


class TestReplyCard:
    def test_success_returns_message_id(self):
        h = _make_handler()
        h.im_client.reply_message.return_value = _success_response("card_msg_001")
        card = json.dumps({"body": {"elements": []}})
        result = h.reply_card("origin_msg", card)
        assert result == "card_msg_001"

    def test_passes_origin_chat_as_audit_alias(self):
        h = _make_handler()
        linker = MessageLinker()
        linker.register_origin(
            "om_card_origin",
            chat_id="oc_requester_dm",
            chat_type="p2p",
            sender_id="ou_requester",
        )
        h.ctx.message_linker = linker
        h.im_client.reply_message.return_value = _success_response("card_msg_001")

        result = h.reply_card("om_card_origin", {"body": {"elements": []}})

        assert result == "card_msg_001"
        assert h.im_client.reply_message.call_args.kwargs["audit_aliases"] == (
            "oc_requester_dm",
            "ou_requester",
        )

    def test_response_failure_returns_none(self):
        h = _make_handler()
        h.im_client.reply_message.return_value = _failure_response()
        card = json.dumps({"body": {"elements": []}})
        result = h.reply_card("origin_msg", card)
        assert result is None

    def test_api_exception_returns_none(self):
        h = _make_handler()
        h.im_client.reply_message.side_effect = Exception("timeout")
        card = json.dumps({"body": {"elements": []}})
        result = h.reply_card("origin_msg", card)
        assert result is None


# ===========================================================================
# update_card tests
# ===========================================================================


class TestUpdateCard:
    def test_success_returns_true(self):
        h = _make_handler()
        h.im_client.patch_message.return_value = _success_response()
        result = h.update_card("msg_id", '{"body":{}}')
        assert result is True

    def test_failure_returns_false(self):
        h = _make_handler()
        h.im_client.patch_message.return_value = _failure_response()
        result = h.update_card("msg_id", '{"body":{}}')
        assert result is False

    def test_exception_returns_false(self):
        h = _make_handler()
        h.im_client.patch_message.side_effect = RuntimeError("boom")
        result = h.update_card("msg_id", '{"body":{}}')
        assert result is False

    def test_passes_canonical_dm_recipient_aliases_for_sibling_patch(self):
        h = _make_handler()
        linker = MessageLinker()
        linker.register_origin(
            "om_card",
            chat_id="oc_requester_dm",
            chat_type="p2p",
            sender_id="ou_requester",
        )
        h.ctx.message_linker = linker
        h.im_client.patch_message.return_value = _success_response()

        assert h.update_card("om_card", '{"body":{}}') is True
        assert h.im_client.patch_message.call_args.kwargs["audit_aliases"] == (
            "oc_requester_dm",
            "ou_requester",
        )


# ===========================================================================
# send_card_to_chat tests
# ===========================================================================


class TestSendCardToChat:
    def test_success_returns_message_id(self):
        h = _make_handler()
        h.im_client.send_message.return_value = _success_response("sent_001")
        card = json.dumps({"body": {"elements": []}})
        result = h.send_card_to_chat("chat_123", card)
        assert result == "sent_001"
        h.im_client.send_message.assert_called_once()

    def test_exception_returns_none(self):
        h = _make_handler()
        h.im_client.send_message.side_effect = Exception("fail")
        result = h.send_card_to_chat("chat_123", "{}")
        assert result is None


# ===========================================================================
# send_text_to_chat tests
# ===========================================================================


class TestSendTextToChat:
    def test_success_returns_message_id(self):
        h = _make_handler()
        h.im_client.send_message.return_value = _success_response("text_001")
        result = h.send_text_to_chat("chat_123", "hello world")
        assert result == "text_001"

    def test_exception_returns_none(self):
        h = _make_handler()
        h.im_client.send_message.side_effect = Exception("fail")
        result = h.send_text_to_chat("chat_123", "hello")
        assert result is None


# ===========================================================================
# _resolve_origin exception fallback
# ===========================================================================


class TestResolveOriginFallback:
    def test_exception_falls_back_to_message_id(self):
        h = _make_handler()
        h.ctx.message_linker.resolve_origin.side_effect = RuntimeError("db error")
        h.im_client.reply_message.return_value = _success_response("msg_ok")
        # Despite _resolve_origin failing, reply_text should still work
        result = h.reply_text("fallback_msg", "test")
        assert result == "msg_ok"


# ===========================================================================
# reply_in_thread default from settings
# ===========================================================================


class TestReplyInThreadDefault:
    def test_thread_mode_passes_true(self):
        h = _make_handler(default_reply_mode="thread")
        h.im_client.reply_message.return_value = _success_response()
        h.reply_text("msg", "hi")
        call_kwargs = h.im_client.reply_message.call_args
        assert call_kwargs[1]["reply_in_thread"] is True

    def test_direct_mode_passes_false(self):
        h = _make_handler(default_reply_mode="direct")
        h.im_client.reply_message.return_value = _success_response()
        h.reply_text("msg", "hi")
        call_kwargs = h.im_client.reply_message.call_args
        assert call_kwargs[1]["reply_in_thread"] is False



# ===========================================================================
# _inject_ref_note post format
# ===========================================================================


class TestInjectRefNoteInteractive:
    def test_ref_note_uses_normal_text_size_for_mobile_readability(self):
        card_content = json.dumps({
            "body": {
                "elements": [
                    {"tag": "markdown", "content": "hello"},
                ],
            },
        })

        result = BaseHandler._inject_ref_note(card_content, "interactive", "REF-001")
        parsed = json.loads(result)
        note = parsed["body"]["elements"][-1]

        assert note["content"] == "REF-001"
        assert note["text_size"] == "normal"


class TestInjectRefNotePost:
    def test_injects_into_last_md_node(self):
        """ref_note should be appended to the last md node's text."""
        post_content = json.dumps({
            "zh_cn": {
                "title": "test",
                "content": [
                    [{"tag": "text", "text": "hello"}],
                    [{"tag": "md", "text": "existing content"}],
                ],
            }
        })
        result = BaseHandler._inject_ref_note(post_content, "post", "REF-001")
        parsed = json.loads(result)
        md_node = parsed["zh_cn"]["content"][1][0]
        assert "REF-001" in md_node["text"]
        assert "existing content" in md_node["text"]

    def test_appends_new_block_when_no_md_node(self):
        """When no md node exists, should append a new block with ref_note."""
        post_content = json.dumps({
            "zh_cn": {
                "title": "test",
                "content": [
                    [{"tag": "text", "text": "only plain text"}],
                ],
            }
        })
        result = BaseHandler._inject_ref_note(post_content, "post", "REF-002")
        parsed = json.loads(result)
        blocks = parsed["zh_cn"]["content"]
        # Should have appended a new block
        assert len(blocks) == 2
        last_block = blocks[-1]
        assert last_block[0]["tag"] == "md"
        assert "REF-002" in last_block[0]["text"]

    def test_empty_ref_note_returns_unchanged(self):
        """Empty ref_note should return content unchanged."""
        post_content = json.dumps({"zh_cn": {"content": [[{"tag": "md", "text": "hi"}]]}})
        result = BaseHandler._inject_ref_note(post_content, "post", "")
        assert result == post_content


# ===========================================================================
# im_client default retry behavior
# ===========================================================================


class TestImClientDefaultRetry:
    def test_reply_message_called_without_max_retries(self):
        """Verify that reply_text does NOT pass max_retries — relying on im_client's built-in default retry."""
        h = _make_handler()
        h.im_client.reply_message.return_value = _success_response()
        h.reply_text("msg", "hello")
        # Ensure max_retries is NOT in the call kwargs (im_client handles retry internally)
        call_args = h.im_client.reply_message.call_args
        assert "max_retries" not in (call_args[1] if call_args[1] else {})

    def test_send_message_called_without_max_retries(self):
        """Verify send_card_to_chat does NOT pass max_retries."""
        h = _make_handler()
        h.im_client.send_message.return_value = _success_response()
        h.send_card_to_chat("chat", '{"body":{}}')
        call_args = h.im_client.send_message.call_args
        assert "max_retries" not in (call_args[1] if call_args[1] else {})
