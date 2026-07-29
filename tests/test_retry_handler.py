"""Unit tests for RetryCommandHandler (src/feishu/retry_handler.py).

Each test class targets one step-method of the handler, verifying it in
isolation from the others.  The mock dispatch object conforms to
``RetryDispatchProtocol`` (public method names only).
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest

from src.utils.signing import sign_command

_SIGNING_KEY = "test_secret_for_retry_handler"


def _compute_hmac_sig(cmd: str) -> str:
    import hmac
    return hmac.new(
        _SIGNING_KEY.encode("utf-8"), cmd.encode("utf-8"), hashlib.sha256,
    ).hexdigest()


def _make_handler():
    """Create a RetryCommandHandler with a mock dispatch (RetryDispatchProtocol)."""
    from src.feishu.retry_handler import RetryCommandHandler, RetryDispatchProtocol
    dispatch = MagicMock(spec=RetryDispatchProtocol)
    dispatch.try_block_with_chat_lock.return_value = False
    dispatch.get_repo_lock_manager.return_value = None
    return RetryCommandHandler(dispatch), dispatch


# ---------------------------------------------------------------------------
# TestVerifySignature
# ---------------------------------------------------------------------------

class TestVerifySignature:
    """_verify_signature: empty sig / wrong sig / correct sig."""

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_empty_sig_replies_expiry(self, _mock_key):
        handler, dispatch = _make_handler()
        result = handler._verify_signature("mid_1", "/deep run", "")
        assert result is False
        dispatch.reply_text.assert_called_once()
        msg = dispatch.reply_text.call_args[0][1]
        assert "失效" in msg

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_wrong_sig_replies_expiry(self, _mock_key):
        handler, dispatch = _make_handler()
        result = handler._verify_signature("mid_2", "/deep run", "deadbeef" * 8)
        assert result is False
        dispatch.reply_text.assert_called_once()
        msg = dispatch.reply_text.call_args[0][1]
        assert "失效" in msg

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_correct_sig_returns_true(self, _mock_key):
        handler, dispatch = _make_handler()
        cmd = "/deep run"
        sig = _compute_hmac_sig(cmd)
        result = handler._verify_signature("mid_3", cmd, sig)
        assert result is True
        dispatch.reply_text.assert_not_called()

    def test_exempt_command_skips_sig_check(self):
        """SIGNATURE_EXEMPT_COMMANDS (e.g. /status) bypass verification."""
        handler, dispatch = _make_handler()
        result = handler._verify_signature("mid_4", "/status", "")
        assert result is True
        dispatch.reply_text.assert_not_called()


# ---------------------------------------------------------------------------
# TestCheckChatLock
# ---------------------------------------------------------------------------

class TestCheckChatLock:
    """_check_chat_lock: blocked / not blocked."""

    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    def test_blocked_returns_true(self, _mock_sender):
        handler, dispatch = _make_handler()
        dispatch.try_block_with_chat_lock.return_value = True
        result = handler._check_chat_lock("chat_1", "mid_1", "/status")
        assert result is True
        dispatch.try_block_with_chat_lock.assert_called_once_with(
            "chat_1", "sender_1", "mid_1", raw_text="/status"
        )

    @patch("src.thread.get_current_sender_id", return_value="sender_2")
    def test_not_blocked_returns_false(self, _mock_sender):
        handler, dispatch = _make_handler()
        dispatch.try_block_with_chat_lock.return_value = False
        result = handler._check_chat_lock("chat_2", "mid_2", "/run")
        assert result is False


# ---------------------------------------------------------------------------
# TestResolveProject
# ---------------------------------------------------------------------------

class TestResolveProject:
    """_resolve_project: pid valid / fallback active / pid mismatch reject."""

    def test_pid_valid_returns_project(self):
        from src.feishu.retry_handler import _REJECTED
        handler, dispatch = _make_handler()
        mock_proj = MagicMock()
        mock_proj.project_id = "proj_1"
        dispatch.get_project_for_chat.return_value = mock_proj

        result = handler._resolve_project("mid_1", "chat_1", "proj_1")
        assert result is mock_proj
        assert result is not _REJECTED

    def test_pid_none_falls_back_to_active(self):
        handler, dispatch = _make_handler()
        mock_proj = MagicMock()
        mock_proj.project_id = "active_proj"
        dispatch.get_active_project.return_value = mock_proj

        result = handler._resolve_project("mid_2", "chat_2", None)
        assert result is mock_proj
        # get_project_for_chat should NOT be called when pid is None
        dispatch.get_project_for_chat.assert_not_called()

    def test_pid_mismatch_returns_rejected(self):
        from src.feishu.retry_handler import _REJECTED
        handler, dispatch = _make_handler()
        # pid="proj_1" but fallback project has different id
        dispatch.get_project_for_chat.return_value = None
        fallback = MagicMock()
        fallback.project_id = "other_proj"
        dispatch.get_active_project.return_value = fallback

        result = handler._resolve_project("mid_3", "chat_3", "proj_1")
        assert result is _REJECTED
        dispatch.reply_text.assert_called_once()
        msg = dispatch.reply_text.call_args[0][1]
        assert "原项目不可用" in msg


# ---------------------------------------------------------------------------
# TestProbeRepoLock
# ---------------------------------------------------------------------------

class TestProbeRepoLock:
    """_probe_repo_lock: success / conflict with handler / conflict without handler / no manager."""

    def test_no_repo_lock_manager_returns_false(self):
        handler, dispatch = _make_handler()
        dispatch.get_repo_lock_manager.return_value = None
        project = MagicMock(root_path="/tmp/repo")
        result = handler._probe_repo_lock("mid_1", "chat_1", "/run", project, 0)
        assert result is False

    def test_probe_success_releases_and_returns_false(self):
        handler, dispatch = _make_handler()
        mock_lock_mgr = MagicMock()
        probe_ok = MagicMock(success=True)
        mock_lock_mgr.acquire.return_value = probe_ok
        dispatch.get_repo_lock_manager.return_value = mock_lock_mgr

        project = MagicMock(root_path="/tmp/repo")
        result = handler._probe_repo_lock("mid_2", "chat_2", "/run", project, 0)

        assert result is False
        mock_lock_mgr.acquire.assert_called_once_with(
            "/tmp/repo",
            "chat_2",
            owner_id=None,
        )
        mock_lock_mgr.release.assert_called_once_with(
            "/tmp/repo",
            "chat_2",
            owner_id=None,
        )

    def test_probe_fail_sends_conflict_card(self):
        handler, dispatch = _make_handler()
        mock_lock_mgr = MagicMock()
        probe_fail = MagicMock(
            success=False, holder_chat_id="other", locked_since=100.0, last_active_time=200.0
        )
        mock_lock_mgr.acquire.return_value = probe_fail
        dispatch.get_repo_lock_manager.return_value = mock_lock_mgr

        project = MagicMock(root_path="/tmp/repo")
        result = handler._probe_repo_lock("mid_3", "chat_3", "/run", project, 1)

        assert result is True
        dispatch.send_lock_conflict_card.assert_called_once()
        assert dispatch.send_lock_conflict_card.call_args.kwargs == {
            "retry_count": 1,
            "chat_id": "chat_3",
        }
        mock_lock_mgr.release.assert_not_called()



# ---------------------------------------------------------------------------
# TestDispatchIntent
# ---------------------------------------------------------------------------

class TestDispatchIntent:
    """_dispatch_intent: normal / LockConflictError / other exception."""

    def test_normal_dispatch(self):
        handler, dispatch = _make_handler()
        project = MagicMock()
        handler._dispatch_intent("mid_1", "chat_1", "/run", project, 0)
        dispatch.process_with_intent.assert_called_once_with("mid_1", "chat_1", "/run", project)

    def test_lock_conflict_sends_card(self):
        from src.repo_lock import LockConflictError
        handler, dispatch = _make_handler()
        dispatch.process_with_intent.side_effect = LockConflictError(
            "conflict", holder_chat_id="other", locked_since=10.0,
            root_path="/tmp", last_active_time=20.0,
        )

        project = MagicMock()
        handler._dispatch_intent("mid_2", "chat_2", "/run", project, 1)

        dispatch.process_with_intent.assert_called_once()
        dispatch.send_lock_conflict_card.assert_called_once()
        assert dispatch.send_lock_conflict_card.call_args.kwargs == {
            "retry_count": 1,
            "chat_id": "chat_2",
        }

    def test_other_exception_reraises(self):
        handler, dispatch = _make_handler()
        dispatch.process_with_intent.side_effect = RuntimeError("boom")
        project = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            handler._dispatch_intent("mid_3", "chat_3", "/run", project, 0)


# ---------------------------------------------------------------------------
# TestRetryDispatchProtocol
# ---------------------------------------------------------------------------

class TestRetryDispatchProtocol:
    """Verify RetryDispatchProtocol runtime_checkable and adapter compliance."""

    def test_adapter_satisfies_protocol(self):
        """_RetryDispatchAdapter must pass isinstance check."""
        from src.feishu.action_registry import _RetryDispatchAdapter
        from src.feishu.retry_handler import RetryDispatchProtocol

        mock_client = MagicMock()
        adapter = _RetryDispatchAdapter(mock_client)
        assert isinstance(adapter, RetryDispatchProtocol)

    def test_adapter_delegates_reply_text(self):
        from src.feishu.action_registry import _RetryDispatchAdapter

        mock_client = MagicMock()
        adapter = _RetryDispatchAdapter(mock_client)
        adapter.reply_text("mid", "hello")
        mock_client._reply_text.assert_called_once_with("mid", "hello")

    def test_adapter_delegates_process_with_intent(self):
        from src.feishu.action_registry import _RetryDispatchAdapter

        mock_client = MagicMock()
        adapter = _RetryDispatchAdapter(mock_client)
        adapter.process_with_intent("mid", "cid", "text", None)
        mock_client._process_with_intent.assert_called_once_with("mid", "cid", "text", None)

    def test_adapter_get_repo_lock_manager_none_ctx(self):
        from src.feishu.action_registry import _RetryDispatchAdapter

        mock_client = MagicMock()
        mock_client._handler_ctx = None
        adapter = _RetryDispatchAdapter(mock_client)
        assert adapter.get_repo_lock_manager() is None

    def test_adapter_send_lock_conflict_card_delegates(self):
        from src.feishu.action_registry import _RetryDispatchAdapter

        mock_client = MagicMock()
        adapter = _RetryDispatchAdapter(mock_client)

        err = MagicMock()
        adapter.send_lock_conflict_card(
            err,
            "mid",
            "/run",
            retry_count=1,
            chat_id="chat_1",
        )
        mock_client.send_lock_conflict_card.assert_called_once_with(
            err,
            "mid",
            "/run",
            retry_count=1,
            chat_id="chat_1",
        )


# ---------------------------------------------------------------------------
# AC-17: Undo lock expiry
# ---------------------------------------------------------------------------


class TestUndoLockExpiry:
    """When undo_lock=True and undo_expires is in the past, reply with expiry message."""

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_expired_undo_replies_expiry_msg(self, _mock_key, tmp_path):
        handler, dispatch = _make_handler()
        val = {
            "command_text": "/unlock",
            "command_sig": sign_command("/unlock", "chat_1"),
            "undo_lock": True,
            "undo_expires": int(time.time()) - 10,  # expired 10s ago
        }
        with patch(
            "src.utils.signing._nonce_store_path",
            return_value=tmp_path / "used-command-nonces.json",
        ):
            handler("mid_undo_1", "chat_1", None, val)
        dispatch.reply_text.assert_called_once()
        msg = dispatch.reply_text.call_args[0][1]
        assert "撤销窗口已关闭" in msg
        # Should NOT dispatch the /unlock command
        dispatch.process_with_intent.assert_not_called()

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    def test_valid_undo_dispatches_unlock(
        self,
        _mock_sender,
        _mock_key,
        tmp_path,
    ):
        handler, dispatch = _make_handler()
        dispatch.get_active_project.return_value = None
        val = {
            "command_text": "/unlock",
            "command_sig": sign_command("/unlock", "chat_2"),
            "undo_lock": True,
            "undo_expires": int(time.time()) + 300,  # still valid
        }
        with patch(
            "src.utils.signing._nonce_store_path",
            return_value=tmp_path / "used-command-nonces.json",
        ):
            handler("mid_undo_2", "chat_2", None, val)
        # Should dispatch the /unlock command
        dispatch.process_with_intent.assert_called_once()
        assert dispatch.process_with_intent.call_args[0][2] == "/unlock"

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    def test_no_undo_lock_field_normal_flow(self, _mock_sender, _mock_key):
        """Without undo_lock, normal retry flow proceeds."""
        handler, dispatch = _make_handler()
        dispatch.get_active_project.return_value = None
        val = {
            "command_text": "/status",
            # /status is SIGNATURE_EXEMPT, no sig needed
        }
        handler("mid_normal", "chat_3", None, val)
        dispatch.process_with_intent.assert_called_once()
