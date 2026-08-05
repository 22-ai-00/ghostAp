"""Unit tests for ChatLockManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.chat_lock import ChatLockCode, ChatLockManager
from src.feishu.slash_command_parser import SlashCommandParser


@pytest.fixture()
def mgr():
    return ChatLockManager()


ADMIN_ID = "admin_001"
NON_ADMIN_ID = "user_002"


def _mock_settings(admin_ids: list[str]):
    """Return a context manager that patches get_settings to include admin_user_ids."""
    mock_settings = MagicMock()
    mock_settings.admin_user_ids = admin_ids
    mock_settings.chat_lock_max_duration = 86400
    mock_settings.chat_lock_cleanup_interval = 60
    return patch("src.chat_lock.get_settings", return_value=mock_settings)


class TestChatLock:

    def test_lock_unlock_chat(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            result = mgr.lock_chat("chat_1", ADMIN_ID)
            assert result.success is True
            assert result.code == ChatLockCode.LOCKED_SUCCESS
            assert mgr.is_locked("chat_1") is True

            result = mgr.unlock_chat("chat_1", ADMIN_ID)
            assert result.success is True
            assert result.code == ChatLockCode.UNLOCKED_SUCCESS
            assert mgr.is_locked("chat_1") is False

    def test_non_admin_cannot_lock(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            result = mgr.lock_chat("chat_1", NON_ADMIN_ID)
            assert result.success is False
            assert result.code == ChatLockCode.CONTACT_ADMIN_TO_LOCK
            assert mgr.is_locked("chat_1") is False

    def test_non_admin_cannot_unlock(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            result = mgr.unlock_chat("chat_1", NON_ADMIN_ID)
            assert result.success is False
            assert result.code == ChatLockCode.CONTACT_ADMIN_UNLOCK
            assert mgr.is_locked("chat_1") is True

    def test_admin_bypass_lock(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            assert mgr.should_block("chat_1", ADMIN_ID) is False

    def test_non_admin_blocked_when_locked(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            assert mgr.should_block("chat_1", NON_ADMIN_ID) is True

    def test_no_block_when_unlocked(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            assert mgr.should_block("chat_1", NON_ADMIN_ID) is False
            assert mgr.should_block("chat_1", ADMIN_ID) is False

    def test_lock_idempotent(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            result = mgr.lock_chat("chat_1", ADMIN_ID)
            assert result.success is True  # already locked, still OK
            assert result.code == ChatLockCode.ALREADY_LOCKED

    def test_unlock_idempotent(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            result = mgr.unlock_chat("chat_1", ADMIN_ID)
            assert result.success is True  # not locked, still OK
            assert result.code == ChatLockCode.NOT_LOCKED

class TestHandleLockCommand:
    """Tests for SystemHandler._handle_lock_command integration."""

    @pytest.fixture(autouse=True)
    def setup_handler(self):
        from src.feishu.handlers.system import SystemHandler

        self.mock_ctx = MagicMock()
        self.mock_ctx.settings.app_id = "test_app"
        self.mock_ctx.settings.app_secret = "test_secret"
        self.mock_ctx.mode_manager.get_mode.return_value = "smart"
        self.mock_ctx.project_manager.get_active_project.return_value = None
        self.mock_ctx.working_dirs = {}

        self.chat_lock_mgr = ChatLockManager()
        self.mock_ctx.chat_lock_manager = self.chat_lock_mgr

        self.handler = SystemHandler(self.mock_ctx)
        self.handler.reply_card = MagicMock()
        self.handler.reply_text = MagicMock()
        self.handler.reply_error = MagicMock()

    def _set_sender(self, sender_id: str):
        from src.thread import set_current_sender_id
        set_current_sender_id(sender_id)

    def test_lock_executes_directly(self):
        """Admin /lock locks immediately (no two-step confirmation)."""
        self.handler.send_card_to_chat = MagicMock()
        with _mock_settings([ADMIN_ID]):
            self._set_sender(ADMIN_ID)
            self.handler._handle_lock_command("msg_1", "chat_1", "lock")
            self._set_sender("")
        # Either reply_card or reply_text should be called once
        assert self.handler.reply_card.call_count + self.handler.reply_text.call_count == 1
        # Chat must be locked immediately
        assert self.chat_lock_mgr.is_locked("chat_1") is True
        # Broadcast card sent to group
        assert self.handler.send_card_to_chat.call_count == 1

    def test_lock_idempotent_shows_status(self):
        """Already-locked chat shows idempotent success card."""
        with _mock_settings([ADMIN_ID]):
            self.chat_lock_mgr.lock_chat("chat_1", ADMIN_ID)
            self._set_sender(ADMIN_ID)
            self.handler._handle_lock_command("msg_1", "chat_1", "lock")
            self._set_sender("")
        self.handler.reply_card.assert_called_once()
        card_content = str(self.handler.reply_card.call_args)
        assert "已处于锁定状态" in card_content

    def test_unlock_by_admin(self):
        with _mock_settings([ADMIN_ID]):
            self.chat_lock_mgr.lock_chat("chat_1", ADMIN_ID)
            self._set_sender(ADMIN_ID)
            self.handler._handle_lock_command("msg_1", "chat_1", "unlock")
            self._set_sender("")
        assert self.handler.reply_card.call_count + self.handler.reply_text.call_count == 1
        assert self.chat_lock_mgr.is_locked("chat_1") is False

    def test_lock_by_non_admin_rejected(self):
        with _mock_settings([ADMIN_ID]):
            self._set_sender(NON_ADMIN_ID)
            self.handler._handle_lock_command("msg_1", "chat_1", "lock")
            self._set_sender("")
        # Non-admin gets a card reply (via reply_card) instead of reply_error
        self.handler.reply_card.assert_called_once()
        assert self.chat_lock_mgr.is_locked("chat_1") is False

    def test_lock_by_non_admin_empty_admin_list(self):
        """When admin_user_ids is empty, show user-friendly guidance (card)."""
        with _mock_settings([]):
            self._set_sender(NON_ADMIN_ID)
            self.handler._handle_lock_command("msg_1", "chat_1", "lock")
            self._set_sender("")
        # Non-admin with no admin config gets card reply
        self.handler.reply_card.assert_called_once()

    def test_unlock_by_non_admin_empty_admin_list(self):
        """When admin_user_ids is empty, /unlock also shows user-friendly guidance (card)."""
        with _mock_settings([]):
            self._set_sender(NON_ADMIN_ID)
            self.handler._handle_lock_command("msg_1", "chat_1", "unlock")
            self._set_sender("")
        # Non-admin with no admin config gets card reply
        self.handler.reply_card.assert_called_once()

    def test_lock_no_sender_id(self):
        with _mock_settings([ADMIN_ID]):
            self._set_sender("")
            self.handler._handle_lock_command("msg_1", "chat_1", "lock")
        self.handler.reply_error.assert_called_once()

    def test_lock_manager_not_configured(self):
        self.mock_ctx.chat_lock_manager = None
        self._set_sender(ADMIN_ID)
        self.handler._handle_lock_command("msg_1", "chat_1", "lock")
        self._set_sender("")
        self.handler.reply_error.assert_called_once()
        # AC-R15: error message should NOT expose internal config key names
        err_msg = self.handler.reply_error.call_args[0][1]
        assert "ADMIN_USER_IDS" not in err_msg
        assert "管理员" in err_msg


class TestIdempotentField:
    """AC-R09: ChatLockResult.idempotent field correctness."""

    def test_lock_idempotent_field(self):
        mgr = ChatLockManager()
        with _mock_settings([ADMIN_ID]):
            result1 = mgr.lock_chat("chat_1", ADMIN_ID)
            assert result1.success is True
            assert result1.idempotent is False
            assert result1.code == ChatLockCode.LOCKED_SUCCESS
            result2 = mgr.lock_chat("chat_1", ADMIN_ID)
            assert result2.success is True
            assert result2.idempotent is True
            assert result2.code == ChatLockCode.ALREADY_LOCKED

    def test_unlock_idempotent_field(self):
        mgr = ChatLockManager()
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            result1 = mgr.unlock_chat("chat_1", ADMIN_ID)
            assert result1.success is True
            assert result1.idempotent is False
            assert result1.code == ChatLockCode.UNLOCKED_SUCCESS
            result2 = mgr.unlock_chat("chat_1", ADMIN_ID)
            assert result2.success is True
            assert result2.idempotent is True
            assert result2.code == ChatLockCode.NOT_LOCKED


class TestReadonlyCommands:
    """Verify READONLY_COMMANDS whitelist allows through even when locked."""

    def test_readonly_commands_not_blocked(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            from src.chat_lock import READONLY_COMMANDS
            for cmd in READONLY_COMMANDS:
                m = SlashCommandParser.parse(cmd)
                assert m is not None
                assert mgr.should_block("chat_1", NON_ADMIN_ID, command_match=m) is False, (
                    f"READONLY command {cmd} should not be blocked"
                )

    def test_command_sets_complete(self):
        from src.chat_lock import READONLY_COMMANDS, SAFE_INTERRUPT_COMMANDS
        # Core readonly commands must be present
        for cmd in ("/status", "/help", "/menu", "/lock", "/unlock"):
            assert cmd in READONLY_COMMANDS, f"{cmd} should be in READONLY_COMMANDS"
        # Safe interrupt commands present
        for cmd in ("/stop_deep", "/stop_spec"):
            assert cmd in SAFE_INTERRUPT_COMMANDS, f"{cmd} should be in SAFE_INTERRUPT_COMMANDS"

    def test_safe_interrupt_commands_not_blocked(self, mgr: ChatLockManager):
        """SAFE_INTERRUPT_COMMANDS bypass lock so users can abort running tasks."""
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            from src.chat_lock import SAFE_INTERRUPT_COMMANDS
            for cmd in SAFE_INTERRUPT_COMMANDS:
                m = SlashCommandParser.parse(cmd)
                assert m is not None
                assert mgr.should_block("chat_1", NON_ADMIN_ID, command_match=m) is False, (
                    f"SAFE_INTERRUPT command {cmd} should not be blocked"
                )

    def test_non_readonly_command_blocked(self, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            m = SlashCommandParser.parse("/run")
            assert m is not None
            assert mgr.should_block("chat_1", NON_ADMIN_ID, command_match=m) is True

    def test_no_command_blocked(self, mgr: ChatLockManager):
        """Regular messages (no command) should be blocked when locked."""
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            assert mgr.should_block("chat_1", NON_ADMIN_ID) is True


class TestGetLockInfo:
    """Tests for ChatLockManager.get_lock_info."""

    def test_get_lock_info_returns_entry_or_none(self, mgr: ChatLockManager):
        info = mgr.get_lock_info("chat_1")
        assert info is None
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            info = mgr.get_lock_info("chat_1")
            assert info is not None
            assert info.locked_by == ADMIN_ID
            assert info.locked_at_wall > 0

    def test_get_lock_info_returns_frozen_copy(self, mgr: ChatLockManager):
        """Returned ChatLockInfo should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            info = mgr.get_lock_info("chat_1")
            assert info is not None
            with pytest.raises(FrozenInstanceError):
                info.locked_by = "hacker"  # type: ignore[misc]


class TestShouldBlockAdminOutsideLock:
    """AC-R16: get_settings() must be called outside self._mu."""

    def test_get_settings_called_before_lock_acquire(self):
        """Verify that admin check (get_settings) runs outside the critical section."""
        mgr = ChatLockManager()
        call_order = []

        original_mu = mgr._mu

        class TrackingLock:
            def __enter__(self_lock):
                call_order.append("lock_acquire")
                return original_mu.__enter__()
            def __exit__(self_lock, *args):
                original_mu.__exit__(*args)
                call_order.append("lock_release")

        def tracked_get_settings():
            call_order.append("get_settings")
            mock_s = MagicMock()
            mock_s.admin_user_ids = [ADMIN_ID]
            return mock_s

        mgr._mu = TrackingLock()
        # Lock the chat first (need real lock for that)
        mgr._mu = original_mu
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
        mgr._mu = TrackingLock()

        with patch("src.chat_lock.get_settings", side_effect=tracked_get_settings):
            mgr.should_block("chat_1", NON_ADMIN_ID)

        # get_settings must appear before lock_acquire
        assert "get_settings" in call_order
        assert "lock_acquire" in call_order
        gs_idx = call_order.index("get_settings")
        la_idx = call_order.index("lock_acquire")
        assert gs_idx < la_idx, f"get_settings ({gs_idx}) should be called before lock_acquire ({la_idx})"


class TestLockConfirmationFlow:
    """Deprecated confirm_lock / cancel_lock callbacks now return deprecation hints."""

    def setup_method(self):
        from unittest.mock import MagicMock

        from src.chat_lock import ChatLockManager
        from src.feishu.handlers.system import SystemHandler

        self.mock_ctx = MagicMock()
        self.mock_ctx.settings.app_id = "test_app"
        self.mock_ctx.settings.app_secret = "test_secret"
        self.mock_ctx.mode_manager.get_mode.return_value = "smart"
        self.mock_ctx.project_manager.get_active_project.return_value = None
        self.mock_ctx.working_dirs = {}

        self.chat_lock_mgr = ChatLockManager()
        self.mock_ctx.chat_lock_manager = self.chat_lock_mgr

        self.handler = SystemHandler(self.mock_ctx)
        self.handler.reply_card = MagicMock()
        self.handler.reply_text = MagicMock()
        self.handler.reply_error = MagicMock()
        self.handler.send_card_to_chat = MagicMock()

    def _set_sender(self, sender_id: str, sender_name: str = ""):
        from src.thread import set_current_sender_id, set_current_sender_name
        set_current_sender_id(sender_id)
        set_current_sender_name(sender_name)

    def test_confirm_and_cancel_lock_return_deprecated_hint(self):
        """handle_confirm_lock and handle_cancel_lock reply with deprecation message."""
        import time
        with _mock_settings([ADMIN_ID]):
            self._set_sender(ADMIN_ID, "Admin")
            self.handler.handle_confirm_lock(
                "msg_1", "chat_1", value={"chat_id": "chat_1", "timestamp": time.time()}
            )
            self._set_sender("")
        # Must NOT lock the chat
        assert self.chat_lock_mgr.is_locked("chat_1") is False
        self.handler.reply_text.assert_called_once()
        reply_text = self.handler.reply_text.call_args[0][1]
        assert "过期" in reply_text or "重新发送" in reply_text

        # Reset and test cancel
        self.handler.reply_text.reset_mock()
        self.handler.handle_cancel_lock("msg_2", "chat_1")
        self.handler.reply_text.assert_called_once()
        cancel_text = self.handler.reply_text.call_args[0][1]
        assert "过期" in cancel_text or "重新发送" in cancel_text


# ---------------------------------------------------------------------------
# READONLY_COMMANDS expansion (Phase-3 F-11)
# ---------------------------------------------------------------------------


class TestReadonlyCommandsExpansion:
    """Verify /帮助, /exit, /quit are in READONLY_COMMANDS and bypass lock."""

    @pytest.mark.parametrize("cmd", ["/帮助", "/exit", "/quit"])
    def test_expanded_commands_in_readonly(self, cmd):
        from src.chat_lock import READONLY_COMMANDS
        assert cmd in READONLY_COMMANDS

    @pytest.mark.parametrize("cmd", ["/帮助", "/exit", "/quit"])
    def test_expanded_commands_bypass_lock(self, cmd, mgr: ChatLockManager):
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            m = SlashCommandParser.parse(cmd)
            assert m is not None
            assert mgr.should_block("chat_1", NON_ADMIN_ID, command_match=m) is False


# ---------------------------------------------------------------------------
# LRU cache placeholder injection (Phase-3 F-09)
# ---------------------------------------------------------------------------


class TestHelpCardLockPlaceholderInjection:
    """Verify that the help card's LRU cache does NOT freeze dynamic lock info."""

    def test_lock_body_not_frozen_across_calls(self):
        """After lock/unlock, the help card must reflect the new state."""
        from src.card.builders.system import _LOCK_BODY_PLACEHOLDER, SystemBuilder

        # The cached version must embed the placeholder, not real lock body
        _msg_type, card_json = SystemBuilder._build_help_card_cached(
            project_name=None,
            root_path=None,
            project_id=None,
            category="main",
            working_dir=None,
            current_mode_str="SMART",
            is_admin=False,
            lock_enabled=True,
        )
        assert _LOCK_BODY_PLACEHOLDER in card_json

    def test_placeholder_replaced_in_build_help_card(self):
        """build_help_card (non-cached wrapper) should replace placeholder with live body."""
        from unittest.mock import MagicMock

        from src.card.builders.system import _LOCK_BODY_PLACEHOLDER, SystemBuilder

        project = MagicMock()
        project.project_name = "test"
        project.root_path = "/tmp/test"
        project.project_id = "pid_1"

        from src.mode import InteractionMode

        # Clear cache to ensure clean state
        SystemBuilder._build_help_card_cached.cache_clear()

        with _mock_settings([ADMIN_ID]):
            _msg_type, card_json = SystemBuilder.build_help_card(
                project=project,
                category="main",
                working_dir="/tmp",
                current_mode=InteractionMode.SMART,
                is_admin=False,
                lock_enabled=True,
                chat_id="chat_placeholder_test",
                session_idle_timeout=600,
                session_idle_warn_at_remaining=120,
                lock_undo_window_seconds=300,
            )
        # Placeholder must be gone; real lock body injected
        assert _LOCK_BODY_PLACEHOLDER not in card_json

        # Clean up cache
        SystemBuilder._build_help_card_cached.cache_clear()


# ---------------------------------------------------------------------------
# Layer-violation guard & format_params tests
# ---------------------------------------------------------------------------


class TestChatLockLayerGuard:
    """Guard test: chat_lock.py must NOT import from src.card (layer violation)."""

    def test_chat_lock_no_ui_dependency(self):
        """AST-level check: src/chat_lock.py must have zero src.card imports."""
        import ast
        import pathlib

        source = pathlib.Path("src/chat_lock.py").read_text()
        tree = ast.parse(source)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "src.card" in node.module:
                violations.append(f"line {node.lineno}: from {node.module} import ...")
        assert violations == [], f"chat_lock.py has UI-layer imports: {violations}"


class TestFormatParams:
    """Verify format_params is populated when needed."""

    def test_contact_named_unlock_has_format_params(self):
        """unlock_chat with locked_by_name should populate format_params['name']."""
        mgr = ChatLockManager()
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID, sender_name="Alice")
            result = mgr.unlock_chat("chat_1", NON_ADMIN_ID)
            assert result.success is False
            assert result.code == ChatLockCode.CONTACT_NAMED_UNLOCK
            assert "name" in result.format_params
            assert result.format_params["name"] == "Alice"

    def test_contact_admin_unlock_and_no_admin_config(self):
        """unlock_chat without locked_by_name has empty format_params; empty admin list returns NO_ADMIN_CONFIG_USER."""
        mgr = ChatLockManager()
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_1", ADMIN_ID)
            result = mgr.unlock_chat("chat_1", NON_ADMIN_ID)
            assert result.success is False
            assert result.code == ChatLockCode.CONTACT_ADMIN_UNLOCK
            assert result.format_params == {}
        mgr2 = ChatLockManager()
        with _mock_settings([]):
            result2 = mgr2.lock_chat("chat_1", NON_ADMIN_ID)
            assert result2.code == ChatLockCode.NO_ADMIN_CONFIG_USER


# ---------------------------------------------------------------------------
# AC-R01: resolve_lock_message testability
# ---------------------------------------------------------------------------
class TestResolveLockMessage:
    """Verify that SystemHandler.resolve_lock_message is independently importable and correct."""

    # Map codes that need format_params to their required params
    _CODES_NEEDING_PARAMS: dict = {
        ChatLockCode.CONTACT_NAMED_UNLOCK: {"name": "Alice"},
    }

    def test_all_codes_return_nonempty(self):
        from src.chat_lock import ChatLockResult
        from src.feishu.handlers.system import SystemHandler
        assert callable(SystemHandler.resolve_lock_message)
        for code in ChatLockCode:
            params = self._CODES_NEEDING_PARAMS.get(code, {})
            result = ChatLockResult(success=True, code=code, format_params=params)
            msg = SystemHandler.resolve_lock_message(result)
            assert isinstance(msg, str) and msg, f"Empty string for {code!r}"

    def test_none_code_and_format_params(self):
        from src.chat_lock import ChatLockResult
        from src.feishu.handlers.system import SystemHandler
        # None code returns empty
        result = ChatLockResult(success=False, code=None)
        assert SystemHandler.resolve_lock_message(result) == ""
        # Format params substitution
        result2 = ChatLockResult(
            success=False,
            code=ChatLockCode.CONTACT_NAMED_UNLOCK,
            format_params={"name": "Alice"},
        )
        msg = SystemHandler.resolve_lock_message(result2)
        assert "Alice" in msg


# ---------------------------------------------------------------------------
# AC-R02: ChatLockCode ↔ UI_TEXT mapping integrity
# ---------------------------------------------------------------------------
class TestChatLockCodeUITextMapping:
    """Ensure every ChatLockCode member has a corresponding non-empty UI_TEXT entry."""

    def test_all_codes_have_nonempty_ui_text(self):
        from src.card.ui_text import UI_TEXT
        missing = [c for c in ChatLockCode if c.value not in UI_TEXT]
        assert not missing, f"ChatLockCode members missing from UI_TEXT: {missing}"
        for code in ChatLockCode:
            val = UI_TEXT[code.value]
            assert isinstance(val, str) and val, (
                f"UI_TEXT[{code.value!r}] should be a non-empty string, got {val!r}"
            )


# ======================================================================
# TTL auto-expiry tests (Task 23-24)
# ======================================================================


class TestChatLockTTLAutoExpiry:
    """ChatLockManager auto-expires locks after max_duration."""

    def test_expired_lock_is_removed(self):
        """Lock older than max_duration is removed by _cleanup_expired."""
        import time as _time
        callbacks: list[tuple[str, object]] = []
        mgr = ChatLockManager(
            max_duration=0.1,  # 100ms
            cleanup_interval=3600,  # don't auto-run
            on_auto_unlock=lambda cid, entry: callbacks.append((cid, entry)),
        )
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_ttl", ADMIN_ID)
            assert mgr.is_locked("chat_ttl")
            _time.sleep(0.15)
            mgr._cleanup_expired()
            assert not mgr.is_locked("chat_ttl")
            assert len(callbacks) == 1
            assert callbacks[0][0] == "chat_ttl"
        mgr.shutdown()

    def test_non_expired_lock_survives(self):
        """Lock younger than max_duration is NOT removed."""
        mgr = ChatLockManager(max_duration=9999, cleanup_interval=3600)
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_live", ADMIN_ID)
            mgr._cleanup_expired()
            assert mgr.is_locked("chat_live")
        mgr.shutdown()

    def test_cleanup_thread_starts_on_lock(self):
        """Cleanup daemon thread is started when a lock is acquired."""
        mgr = ChatLockManager(max_duration=86400, cleanup_interval=60)
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_daemon", ADMIN_ID)
            assert mgr._cleanup_thread is not None
            assert mgr._cleanup_thread.is_alive()
        mgr.shutdown()

    def test_shutdown_stops_cleanup_thread(self):
        """shutdown() stops the cleanup thread."""
        mgr = ChatLockManager(max_duration=86400, cleanup_interval=60)
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_stop", ADMIN_ID)
        mgr.shutdown()
        assert mgr._cleanup_thread is None or not mgr._cleanup_thread.is_alive()

    def test_on_auto_unlock_callback_receives_entry(self):
        """on_auto_unlock callback receives (chat_id, ChatLockEntry)."""
        import time as _time

        from src.chat_lock import ChatLockEntry
        received = []
        mgr = ChatLockManager(
            max_duration=0.05,
            cleanup_interval=3600,
            on_auto_unlock=lambda cid, e: received.append((cid, e)),
        )
        with _mock_settings([ADMIN_ID]):
            mgr.lock_chat("chat_cb", ADMIN_ID, sender_name="TestAdmin")
            _time.sleep(0.1)
            mgr._cleanup_expired()
        assert len(received) == 1
        cid, entry = received[0]
        assert cid == "chat_cb"
        assert isinstance(entry, ChatLockEntry)
        assert entry.locked_by == ADMIN_ID
        assert entry.locked_by_name == "TestAdmin"
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Task 26: TestHandleForceReleaseRepoLock
# ---------------------------------------------------------------------------


class TestHandleForceReleaseRepoLock:
    """AC-37: handle_force_release_repo_lock permission + error paths."""

    @pytest.fixture(autouse=True)
    def setup_handler(self):
        from src.feishu.handlers.system import SystemHandler

        self.mock_ctx = MagicMock()
        self.mock_ctx.settings.app_id = "test_app"
        self.mock_ctx.settings.app_secret = "test_secret"
        self.mock_ctx.mode_manager.get_mode.return_value = "smart"
        self.mock_ctx.project_manager.get_active_project.return_value = None
        self.mock_ctx.working_dirs = {}

        self.chat_lock_mgr = ChatLockManager()
        self.mock_ctx.chat_lock_manager = self.chat_lock_mgr

        self.handler = SystemHandler(self.mock_ctx)
        self.handler.reply_card = MagicMock()
        self.handler.reply_text = MagicMock()
        self.handler.reply_error = MagicMock()

    def _set_sender(self, sender_id: str):
        from src.thread import set_current_sender_id
        set_current_sender_id(sender_id)

    def test_permission_denied_paths(self):
        """chat_lock_manager=None or non-admin triggers permission denied."""
        # CLM None path
        self.mock_ctx.chat_lock_manager = None
        self._set_sender(ADMIN_ID)
        self.handler.handle_force_release_repo_lock("msg_1", "chat_1")
        self.handler.reply_error.assert_called_once()

        # Non-admin path
        self.handler.reply_error.reset_mock()
        self.mock_ctx.chat_lock_manager = self.chat_lock_mgr
        with _mock_settings([ADMIN_ID]):
            self._set_sender(NON_ADMIN_ID)
            self.handler.handle_force_release_repo_lock("msg_2", "chat_1")
        self.handler.reply_error.assert_called_once()

    def test_missing_repo_lock_mgr(self):
        """repo_lock_manager=None → error reply."""
        self.mock_ctx.repo_lock_manager = None
        with _mock_settings([ADMIN_ID]):
            self._set_sender(ADMIN_ID)
            self.handler.handle_force_release_repo_lock("msg_1", "chat_1")
        self.handler.reply_error.assert_called_once()

    def test_missing_root_path_in_value(self):
        """No root_path or repo_token → error reply."""
        from src.repo_lock import RepoLockManager
        repo_mgr = RepoLockManager(idle_timeout=999, cleanup_interval=999)
        self.mock_ctx.repo_lock_manager = repo_mgr
        try:
            with _mock_settings([ADMIN_ID]):
                self._set_sender(ADMIN_ID)
                self.handler.handle_force_release_repo_lock("msg_1", "chat_1", value={})
            self.handler.reply_error.assert_called_once()
        finally:
            repo_mgr.shutdown()


# ---------------------------------------------------------------------------
# Task 28: TestLockGroupNotification
# ---------------------------------------------------------------------------


class TestLockGroupNotification:
    """AC-26: Non-idempotent lock broadcasts group notification via send_message."""

    @pytest.fixture(autouse=True)
    def setup_handler(self):
        from src.feishu.handlers.system import SystemHandler

        self.mock_ctx = MagicMock()
        self.mock_ctx.settings.app_id = "test_app"
        self.mock_ctx.settings.app_secret = "test_secret"
        self.mock_ctx.settings.feishu_app_id = "cli_app"
        self.mock_ctx.mode_manager.get_mode.return_value = "smart"
        self.mock_ctx.project_manager.get_active_project.return_value = None
        self.mock_ctx.working_dirs = {}

        self.chat_lock_mgr = ChatLockManager()
        self.mock_ctx.chat_lock_manager = self.chat_lock_mgr

        self.handler = SystemHandler(self.mock_ctx)
        self.handler.reply_card = MagicMock()
        self.handler.reply_text = MagicMock()
        self.handler.reply_error = MagicMock()
        self.handler.send_card_to_chat = MagicMock()

    def test_lock_broadcasts_group_notification(self):
        """First lock triggers send_message broadcast."""
        from src.thread import set_current_sender_id, set_current_sender_name
        set_current_sender_id(ADMIN_ID)
        set_current_sender_name("Alice")
        with _mock_settings([ADMIN_ID]):
            self.handler._handle_lock_command("msg_1", "chat_1", "lock")
        # reply_card or reply_text is the personal reply; _send_card_to_chat is the broadcast
        assert self.handler.reply_card.call_count + self.handler.reply_text.call_count == 1
        assert self.handler.send_card_to_chat.call_count == 1

    def test_idempotent_lock_no_broadcast(self):
        """Already-locked → idempotent reply, no broadcast."""
        from src.thread import set_current_sender_id, set_current_sender_name
        set_current_sender_id(ADMIN_ID)
        set_current_sender_name("Alice")
        with _mock_settings([ADMIN_ID]):
            self.chat_lock_mgr.lock_chat("chat_1", ADMIN_ID, sender_name="Alice")
            self.handler._handle_lock_command("msg_2", "chat_1", "lock")
        self.handler.reply_card.assert_called_once()
        self.handler.send_card_to_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Task 31: TestChatLockConcurrency
# ---------------------------------------------------------------------------


class TestChatLockConcurrency:
    """AC-39: ChatLockManager thread-safety under concurrent lock/unlock."""

    def test_concurrent_lock_unlock(self):
        """ThreadPoolExecutor(10) concurrent lock/unlock — no exceptions, consistent state."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        mgr = ChatLockManager()
        errors = []

        def _worker(i: int):
            try:
                with _mock_settings([ADMIN_ID]):
                    r = mgr.lock_chat(f"chat_{i % 3}", ADMIN_ID, sender_name="Worker")
                    if r.success:
                        mgr.unlock_chat(f"chat_{i % 3}", ADMIN_ID)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_worker, i) for i in range(50)]
            for f in as_completed(futures):
                f.result()

        assert errors == []
        mgr.shutdown()


class TestGetAllowedCommandsDisplay:
    """F-05: ChatLockManager.get_allowed_commands_display() — single source of truth."""

    def test_returns_nonempty_string(self):
        result = ChatLockManager.get_allowed_commands_display()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_content_correctness(self):
        result = ChatLockManager.get_allowed_commands_display()
        # Excludes admin commands
        assert "`/lock`" not in result
        assert "`/unlock`" not in result
        # Includes status and help
        assert "`/status`" in result
        assert "`/help`" in result
        # Includes safe interrupt commands
        from src.chat_lock import SAFE_INTERRUPT_COMMANDS
        for cmd in SAFE_INTERRUPT_COMMANDS:
            assert f"`{cmd}`" in result

    def test_sorted_alphabetically(self):
        """Commands appear in sorted order."""
        import re
        result = ChatLockManager.get_allowed_commands_display()
        cmds = re.findall(r"`(/[^`]+)`", result)
        assert cmds == sorted(cmds)


class TestForceReleaseHolderRaceGuard:
    """F-01: handle_confirm_force_release verifies _hcid before releasing."""

    @pytest.fixture(autouse=True)
    def setup_handler(self):
        from src.feishu.handlers.system import SystemHandler

        self.mock_ctx = MagicMock()
        self.mock_ctx.settings.app_id = "test_app"
        self.mock_ctx.settings.app_secret = "test_secret"
        self.mock_ctx.settings.lock_confirm_timeout = 120
        self.mock_ctx.mode_manager.get_mode.return_value = "smart"
        self.mock_ctx.project_manager.get_active_project.return_value = None
        self.mock_ctx.working_dirs = {}

        self.chat_lock_mgr = ChatLockManager()
        self.mock_ctx.chat_lock_manager = self.chat_lock_mgr

        self.repo_lock_mgr = MagicMock()
        self.mock_ctx.repo_lock_manager = self.repo_lock_mgr

        self.handler = SystemHandler(self.mock_ctx)
        self.handler.reply_card = MagicMock()
        self.handler.reply_text = MagicMock()
        self.handler.reply_error = MagicMock()
        self.handler.send_card_to_chat = MagicMock()

    def _set_sender(self, sender_id: str):
        from src.thread import set_current_sender_id
        set_current_sender_id(sender_id)

    def test_hcid_match_allows_release(self):
        """When _hcid matches current holder, force_release proceeds."""
        import time as _t

        lock_info = MagicMock()
        lock_info.chat_id = "holder_chat_A"
        self.repo_lock_mgr.get_lock_info.return_value = lock_info
        self.repo_lock_mgr.token_to_path.return_value = "/tmp/repo"

        with _mock_settings([ADMIN_ID]):
            self._set_sender(ADMIN_ID)
            self.handler.handle_confirm_force_release(
                "msg_1", "chat_admin",
                value={
                    "_tk": "tok_1",
                    "_ts": _t.time(),
                    "_hcid": "holder_chat_A",
                },
            )

        self.repo_lock_mgr.force_release.assert_called_once_with("/tmp/repo")
        self.handler.reply_error.assert_not_called()

    def test_hcid_mismatch_blocks_release(self):
        """When _hcid doesn't match current holder, force_release is blocked."""
        import time as _t

        lock_info = MagicMock()
        lock_info.chat_id = "holder_chat_B"  # different from _hcid
        self.repo_lock_mgr.get_lock_info.return_value = lock_info
        self.repo_lock_mgr.token_to_path.return_value = "/tmp/repo"

        with _mock_settings([ADMIN_ID]):
            self._set_sender(ADMIN_ID)
            self.handler.handle_confirm_force_release(
                "msg_1", "chat_admin",
                value={
                    "_tk": "tok_1",
                    "_ts": _t.time(),
                    "_hcid": "holder_chat_A",  # stale — holder has changed
                },
            )

        self.repo_lock_mgr.force_release.assert_not_called()
        self.handler.reply_error.assert_called_once()
        # Verify error message mentions holder change
        error_msg = self.handler.reply_error.call_args[0][1]
        assert "变化" in error_msg

    def test_no_hcid_backward_compat(self):
        """Old cards without _hcid still allow force_release (backward compat)."""
        import time as _t

        lock_info = MagicMock()
        lock_info.chat_id = "holder_chat_X"
        self.repo_lock_mgr.get_lock_info.return_value = lock_info
        self.repo_lock_mgr.token_to_path.return_value = "/tmp/repo"

        with _mock_settings([ADMIN_ID]):
            self._set_sender(ADMIN_ID)
            self.handler.handle_confirm_force_release(
                "msg_1", "chat_admin",
                value={
                    "_tk": "tok_1",
                    "_ts": _t.time(),
                    # no _hcid — old card format
                },
            )

        self.repo_lock_mgr.force_release.assert_called_once_with("/tmp/repo")
        self.handler.reply_error.assert_not_called()


# ======================================================================
# FS-21: should_block_card_action exhaustive tests
# ======================================================================


class TestShouldBlockCardActionExhaustive:
    """Exhaustive tests for ChatLockManager.should_block_card_action."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with _mock_settings([ADMIN_ID]):
            self.mgr = ChatLockManager()
            self.mgr.lock_chat("chat1", ADMIN_ID)
        yield
        self.mgr.shutdown()

    def test_not_locked_allows_all(self):
        with _mock_settings([ADMIN_ID]):
            mgr = ChatLockManager()
        assert mgr.should_block_card_action("unlocked_chat", NON_ADMIN_ID, "dangerous_action") is False

    def test_admin_allowed_through(self):
        with _mock_settings([ADMIN_ID]):
            assert self.mgr.should_block_card_action("chat1", ADMIN_ID, "dangerous_action") is False

    def test_non_admin_blocked(self):
        with _mock_settings([ADMIN_ID]):
            assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, "enter_coco") is True

    def test_stop_suffix_exempt(self):
        with _mock_settings([ADMIN_ID]):
            assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, "deep_stop") is False
            assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, "spec_stop") is False

    def test_show_prefix_exempt(self):
        with _mock_settings([ADMIN_ID]):
            assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, "show_status") is False
            assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, "show_board") is False

    def test_card_exempt_actions(self):
        with _mock_settings([ADMIN_ID]):
            for action in ChatLockManager.CARD_EXEMPT_ACTIONS:
                assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, action) is False, \
                    f"CARD_EXEMPT_ACTIONS entry {action!r} was blocked"

    def test_readonly_commands_exempt(self):
        from src.chat_lock import READONLY_COMMANDS
        with _mock_settings([ADMIN_ID]):
            for cmd in READONLY_COMMANDS:
                assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, cmd) is False

    def test_safe_interrupt_commands_exempt(self):
        from src.chat_lock import SAFE_INTERRUPT_COMMANDS
        with _mock_settings([ADMIN_ID]):
            for cmd in SAFE_INTERRUPT_COMMANDS:
                assert self.mgr.should_block_card_action("chat1", NON_ADMIN_ID, cmd) is False


# ======================================================================
# FS-23: ChatLock invalid chat_id tests
# ======================================================================


class TestChatLockInvalidChatId:
    """Edge cases for empty / None chat_id arguments."""

    def test_empty_chat_id_operations(self):
        with _mock_settings([ADMIN_ID]):
            mgr = ChatLockManager()
            result = mgr.lock_chat("", ADMIN_ID)
            assert result.success is True
            mgr.shutdown()

        with _mock_settings([ADMIN_ID]):
            mgr2 = ChatLockManager()
        assert mgr2.is_locked("") is False
        assert mgr2.should_block("", NON_ADMIN_ID) is False

    def test_unlock_never_locked(self):
        with _mock_settings([ADMIN_ID]):
            mgr = ChatLockManager()
            result = mgr.unlock_chat("never_locked_chat", ADMIN_ID)
            # Idempotent: unlocking a never-locked chat succeeds with NOT_LOCKED
            assert result.success is True
            assert result.code == ChatLockCode.NOT_LOCKED
            assert result.idempotent is True


class TestLockChatAfterShutdown:
    """AC-26: lock_chat() after shutdown() must not crash and return a clear result."""

    def test_lock_and_unlock_after_shutdown(self):
        m = ChatLockManager(max_duration=86400, cleanup_interval=60)
        m.shutdown()

        # lock_chat after shutdown should not crash
        with _mock_settings([ADMIN_ID]):
            result = m.lock_chat("chat_post_shutdown", ADMIN_ID, sender_name="Admin")
        assert result.success is True or result.code is not None

        # unlock after shutdown should still work
        m2 = ChatLockManager(max_duration=86400, cleanup_interval=60)
        with _mock_settings([ADMIN_ID]):
            m2.lock_chat("chat_x", ADMIN_ID)
        m2.shutdown()
        with _mock_settings([ADMIN_ID]):
            result2 = m2.unlock_chat("chat_x", ADMIN_ID)
        assert result2.success is True

    def test_should_block_after_shutdown(self):
        m = ChatLockManager(max_duration=86400, cleanup_interval=60)
        with _mock_settings([ADMIN_ID]):
            m.lock_chat("chat_y", ADMIN_ID)
        m.shutdown()

        # should_block must still work
        with _mock_settings([ADMIN_ID]):
            assert m.should_block("chat_y", "non_admin") is True
            assert m.should_block("chat_y", ADMIN_ID) is False


# ---------------------------------------------------------------------------
# Task 37: shutdown_if_active idempotent module-level function
# ---------------------------------------------------------------------------


class TestShutdownIfActive:
    """shutdown_if_active() is idempotent and safe when no instance exists."""

    def test_noop_and_idempotent(self):
        from src import chat_lock as _mod
        _orig = _mod._instance
        try:
            _mod._instance = None
            _mod.shutdown_if_active()  # should not raise

            m = ChatLockManager(max_duration=86400, cleanup_interval=60)
            _mod._instance = m
            _mod.shutdown_if_active()
            _mod.shutdown_if_active()  # second call should not raise
        finally:
            _mod._instance = _orig
