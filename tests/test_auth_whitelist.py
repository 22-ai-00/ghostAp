"""Tests for authorization whitelist (security hardening A2).

Covers:
- normal messages require both an enrolled user/admin and an enrolled chat.
- empty allowlists deny normal messages.
- denied messages stop before every business side effect.
- Config parsing: comma-separated string -> frozenset.
"""

import hashlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import (
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    IngressAccessRequest,
)
from src.config import IngressAccessMode
from src.config.settings import Settings

# ==================================================================
# Config parsing tests
# ==================================================================


class TestWhitelistConfigParsing:
    """Test allowed_chat_ids / allowed_user_ids coerce from string to frozenset."""

    def test_empty_string_yields_empty_frozenset(self):
        s = Settings(allowed_chat_ids="", allowed_user_ids="")
        assert s.allowed_chat_ids == frozenset()
        assert s.allowed_user_ids == frozenset()

    def test_single_value(self):
        s = Settings(allowed_chat_ids="chat_abc", allowed_user_ids="user_xyz")
        assert s.allowed_chat_ids == frozenset({"chat_abc"})
        assert s.allowed_user_ids == frozenset({"user_xyz"})

    def test_comma_separated_values(self):
        s = Settings(
            allowed_chat_ids="chat_1, chat_2,chat_3",
            allowed_user_ids="u1,u2, u3 ",
        )
        assert s.allowed_chat_ids == frozenset({"chat_1", "chat_2", "chat_3"})
        assert s.allowed_user_ids == frozenset({"u1", "u2", "u3"})

    def test_list_input_normalized(self):
        s = Settings(
            allowed_chat_ids=["c1", "c2"],  # type: ignore[arg-type]
            allowed_user_ids=frozenset({"u1"}),  # type: ignore[arg-type]
        )
        assert s.allowed_chat_ids == frozenset({"c1", "c2"})
        assert s.allowed_user_ids == frozenset({"u1"})

    def test_whitespace_only_yields_empty(self):
        s = Settings(allowed_chat_ids="  ,  , ", allowed_user_ids=" ")
        assert s.allowed_chat_ids == frozenset()
        assert s.allowed_user_ids == frozenset()


# ==================================================================
# Whitelist enforcement tests (ws_client._process_message_async)
# ==================================================================


def _make_fake_data(chat_id: str = "oc_ok", sender_id: str = "ou_ok"):
    """Build a minimal mock P2ImMessageReceiveV1 for _process_message_async."""
    data = MagicMock()
    data.event.message.message_id = "om_001"
    data.event.message.chat_id = chat_id
    data.event.message.chat_type = "group"
    data.event.message.create_time = "9999999999999"
    data.event.message.message_type = "text"
    data.event.message.content = '{"text": "hello"}'
    data.event.message.parent_id = None
    data.event.message.root_id = None
    data.event.sender.sender_id.open_id = sender_id
    return data


@pytest.fixture
def _patch_settings():
    """Provide a helper to patch get_settings with custom whitelist values."""

    def _factory(allowed_chat_ids: str = "", allowed_user_ids: str = ""):
        s = Settings(
            allowed_chat_ids=allowed_chat_ids,
            allowed_user_ids=allowed_user_ids,
        )
        return s

    return _factory


class TestWhitelistEnforcement:
    """Integration-level tests: messages are dropped or passed through based on whitelist."""

    @patch("src.feishu.ws_client.FeishuWSClient.__init__", return_value=None)
    def _build_client(self, mock_init):
        from src.feishu.ws_client import FeishuWSClient

        client = FeishuWSClient.__new__(FeishuWSClient)
        return client

    def _setup_client(self, settings):
        """Set up a minimal client with mocked internals for testing _process_message_async."""
        client = self._build_client()
        client.settings = settings
        client._message_ingress_guard = MagicMock()
        client._message_ingress_guard.is_message_expired.return_value = False
        client._message_ingress_guard.is_duplicate_message.return_value = False
        client._message_cache = MagicMock()
        client._message_cache.is_duplicate.return_value = False
        client._get_image_handler = MagicMock()
        parse_result = MagicMock()
        parse_result.text = "hello"
        parse_result.image_keys = []
        client._get_image_handler.return_value.parse_message.return_value = parse_result
        client._chat_lock_gate = MagicMock()
        client._chat_lock_gate.check.return_value = False
        client._pending_image_lock = MagicMock()
        client._pending_image_lock.__enter__ = MagicMock(return_value=None)
        client._pending_image_lock.__exit__ = MagicMock(return_value=False)
        client._pending_image_keys = {}
        client._pending_image_only = set()
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = None
        client._project_manager = MagicMock()
        client._project_manager.find_by_bound_chat_id.return_value = None
        client._project_manager.get_active_project.return_value = None
        client._message_mapper = MagicMock()
        client._message_mapper.get_project_id.return_value = None
        client._scheduler = MagicMock()
        client._message_linker = MagicMock()
        client._mode_manager = MagicMock()
        client._image_handler = None
        client._enable_streaming = False
        client._employee_department_runtime = MagicMock()
        client._handle_image_content = MagicMock()
        client._is_likely_shell_command_message = MagicMock(return_value=False)
        # Router-bound methods (normally attached by bind_forwarding_methods)
        client._ensure_request_id = MagicMock(return_value="req_test_001")
        client._get_api_client = MagicMock()
        # Mock _dispatch_message_logic to track whether it was called
        client._dispatch_message_logic = MagicMock()
        client._show_help = MagicMock()
        client._reply_text = MagicMock()
        client._dispatch_empty_text = MagicMock()
        return client

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_chat_not_in_whitelist_dropped(self, _mock_resolve, _patch_settings):
        """Message from non-whitelisted chat is silently dropped."""
        settings = _patch_settings(allowed_chat_ids="oc_allowed")
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_blocked", sender_id="ou_ok")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_user_not_in_whitelist_dropped(self, _mock_resolve, _patch_settings):
        """Message from non-whitelisted user is silently dropped."""
        settings = _patch_settings(allowed_user_ids="ou_allowed")
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_ok", sender_id="ou_blocked")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_chat_only_enrolment_is_denied(self, _mock_resolve, _patch_settings):
        """An enrolled chat without an enrolled user still fails closed."""
        settings = _patch_settings(allowed_chat_ids="oc_ok,oc_other")
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_ok", sender_id="ou_ok")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_user_only_enrolment_is_denied(self, _mock_resolve, _patch_settings):
        """An enrolled user without an enrolled chat still fails closed."""
        settings = _patch_settings(allowed_user_ids="ou_ok")
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_ok", sender_id="ou_ok")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_both_empty_denies_all(self, _mock_resolve, _patch_settings):
        """Empty allowlists deny normal messages by default."""
        settings = _patch_settings(allowed_chat_ids="", allowed_user_ids="")
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_any", sender_id="ou_any")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_both_whitelists_enforced(self, _mock_resolve, _patch_settings):
        """Both whitelists are checked: chat passes but user fails -> dropped."""
        settings = _patch_settings(
            allowed_chat_ids="oc_ok",
            allowed_user_ids="ou_allowed",
        )
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_ok", sender_id="ou_blocked")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name", return_value="TestUser")
    def test_both_whitelists_pass(self, _mock_resolve, _patch_settings):
        """Both whitelists pass: message goes through."""
        settings = _patch_settings(
            allowed_chat_ids="oc_ok",
            allowed_user_ids="ou_ok",
        )
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_ok", sender_id="ou_ok")

        client._process_message_async(data)

        client._dispatch_message_logic.assert_called_once()

    @patch("src.feishu.user_cache.resolve_display_name_nonblocking")
    def test_denied_async_message_has_zero_business_side_effects(
        self,
        mock_resolve_name,
        _patch_settings,
    ):
        settings = _patch_settings(
            allowed_chat_ids="oc_allowed",
            allowed_user_ids="ou_allowed",
        )
        client = self._setup_client(settings)
        parse_result = client._get_image_handler.return_value.parse_message.return_value
        parse_result.image_keys = ["img_1"]
        data = _make_fake_data(chat_id="oc_blocked", sender_id="ou_blocked")

        client._process_message_async(data)

        mock_resolve_name.assert_not_called()
        client._chat_lock_gate.check.assert_not_called()
        client._employee_department_runtime.record_group_event.assert_not_called()
        client._handle_image_content.assert_not_called()
        client._dispatch_message_logic.assert_not_called()

    def test_denied_ingress_never_reaches_scheduler_shell_or_linker(
        self,
        _patch_settings,
    ):
        settings = _patch_settings(
            allowed_chat_ids="oc_allowed",
            allowed_user_ids="ou_allowed",
        )
        client = self._setup_client(settings)
        data = _make_fake_data(chat_id="oc_blocked", sender_id="ou_blocked")

        client._handle_message(data)

        client._scheduler.submit.assert_not_called()
        client._is_likely_shell_command_message.assert_not_called()
        client._message_linker.register_origin.assert_not_called()

    @patch("src.feishu.user_cache.resolve_display_name_nonblocking")
    def test_current_chat_enrolment_bypasses_group_ledger_and_images(
        self,
        mock_resolve_name,
        _patch_settings,
    ):
        settings = _patch_settings()
        object.__setattr__(settings, "admin_user_ids", frozenset({"ou_ok"}))
        client = self._setup_client(settings)
        parse_result = client._get_image_handler.return_value.parse_message.return_value
        parse_result.text = "/access allow-chat"
        parse_result.image_keys = ["img_ignored"]
        client._system_handler = MagicMock()
        data = _make_fake_data(chat_id="oc_new", sender_id="ou_ok")

        client._process_message_async(data)

        client._system_handler.handle_intercepted_command.assert_called_once()
        call = client._system_handler.handle_intercepted_command.call_args
        assert call.args[:3] == (
            "om_001",
            "oc_new",
            "/access allow-chat",
        )
        assert call.kwargs["command_match"].command == "/access"
        client._employee_department_runtime.record_group_event.assert_not_called()
        client._handle_image_content.assert_not_called()
        client._chat_lock_gate.check.assert_not_called()
        mock_resolve_name.assert_not_called()

    def test_shadow_audit_contains_only_hashed_identifiers(
        self,
        _patch_settings,
        caplog,
    ):
        settings = _patch_settings()
        object.__setattr__(settings, "ingress_access_mode", "shadow")
        client = self._setup_client(settings)
        sender_id = "ou_sensitive_sender"
        chat_id = "oc_sensitive_chat"
        caplog.set_level(logging.WARNING, logger="ghostap.audit")

        decision = client._decide_ingress_access(
            message_id="om_sensitive_message",
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type="group",
            command_match=None,
        )

        assert decision.allowed is True
        assert decision.prospective_allowed is False
        assert sender_id not in caplog.text
        assert chat_id not in caplog.text
        assert hashlib.sha256(sender_id.encode()).hexdigest()[:16] in caplog.text
        assert hashlib.sha256(chat_id.encode()).hexdigest()[:16] in caplog.text

    def test_worker_reauthorizes_against_current_policy_snapshot(
        self,
        _patch_settings,
    ):
        settings = _patch_settings(
            allowed_chat_ids="oc_ok",
            allowed_user_ids="ou_ok",
        )
        client = self._setup_client(settings)
        old_policy = IngressAccessPolicy(
            admin_ids=frozenset(),
            allowed_user_ids=frozenset({"ou_ok"}),
            allowed_chat_ids=frozenset({"oc_ok"}),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
        provider = IngressAccessPolicyProvider(old_policy)
        client._ingress_access_policy_provider = provider
        request = IngressAccessRequest(
            message_id="om_001",
            sender_id="ou_ok",
            chat_id="oc_ok",
            chat_type="group",
            command_match=None,
        )
        assert old_policy.decide(request).allowed is True
        provider.swap(
            IngressAccessPolicy(
                admin_ids=frozenset(),
                allowed_user_ids=frozenset(),
                allowed_chat_ids=frozenset(),
                mode=IngressAccessMode.ENFORCED,
                admin_bootstrap_scope="p2p_only",
            )
        )
        data = _make_fake_data(chat_id="oc_ok", sender_id="ou_ok")

        client._process_message_async(data)

        client._employee_department_runtime.record_group_event.assert_not_called()
        client._handle_image_content.assert_not_called()
        client._dispatch_message_logic.assert_not_called()
