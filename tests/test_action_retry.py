"""Integration tests for _handle_retry_command (src/feishu/action_registry.py).

Covers: signature verification, UI_TEXT wording, chat lock check,
repo lock probe-acquire-then-release, and TOCTOU race documentation.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — minimal mock client + registry bootstrap
# ---------------------------------------------------------------------------

_SIGNING_KEY = "test_secret_for_retry"


def _make_mock_client():
    """Build a minimal mock FeishuWSClient that init_action_registry can accept."""
    client = MagicMock()
    # _register_action: capture handlers by exact action name
    _handlers: dict[str, object] = {}

    def _register_action(handler, exact=None, prefix=None):
        if exact:
            _handlers[exact] = handler

    client._register_action = _register_action
    client._handlers = _handlers
    # Defaults that _handle_retry_command reads
    client._chat_lock_gate = MagicMock()
    client._chat_lock_gate.check = MagicMock(return_value=False)
    client._chat_lock_gate.check_card_action = MagicMock(return_value=False)
    client._handler_ctx = None  # no repo lock by default
    return client


def _bootstrap(client):
    """Run init_action_registry and return the retry_command handler.

    MagicMock auto-creates missing attributes, so register_programming_mode_actions
    can run without real handler methods on the client.
    """
    from src.feishu.action_registry import init_action_registry
    init_action_registry(client)
    return client._handlers.get("retry_command")


def _sign_command(cmd: str, chat_id: str) -> str:
    """Create the production v2 signature bound to the action's chat."""
    from src.utils.signing import sign_command

    return sign_command(cmd, chat_id)


@pytest.fixture(autouse=True)
def _isolated_nonce_store(tmp_path, monkeypatch):
    """Keep durable single-use action state isolated between tests."""
    from src.utils import signing

    signing._USED_NONCES.clear()
    monkeypatch.setattr(
        signing,
        "_nonce_store_path",
        lambda: tmp_path / "used-command-nonces.json",
    )
    yield
    signing._USED_NONCES.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetryCommandEmptySig:
    """AC-21: Empty or missing command_sig → friendly expiry message."""

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_empty_sig_returns_expiry_message(self, _mock_key):
        client = _make_mock_client()
        handler = _bootstrap(client)
        assert handler is not None

        handler("mid_1", "chat_1", None, {"command_text": "/deep run"})
        client._reply_text.assert_called_once()
        msg = client._reply_text.call_args[0][1]
        assert "失效" in msg

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_missing_sig_key_exempt_command_proceeds(self, _mock_key):
        """SIGNATURE_EXEMPT_COMMANDS: /status with empty sig is NOT rejected."""
        client = _make_mock_client()
        handler = _bootstrap(client)

        handler("mid_2", "chat_2", None, {"command_text": "/status", "command_sig": ""})
        # /status is exempt — should NOT get an expiry reply
        for call in client._reply_text.call_args_list:
            msg = call[0][1] if call[0] else ""
            assert "失效" not in msg


class TestRetryCommandWrongSig:
    """AC-22: Invalid signature → friendly expiry message."""

    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_wrong_sig_returns_expiry_message(self, _mock_key):
        client = _make_mock_client()
        handler = _bootstrap(client)

        handler("mid_3", "chat_3", None, {
            "command_text": "/deep run",
            "command_sig": "deadbeef" * 8,
        })
        client._reply_text.assert_called_once()
        msg = client._reply_text.call_args[0][1]
        assert "失效" in msg


class TestRetryCommandValidSig:
    """AC-23: Correct HMAC sig → dispatches to _process_with_intent."""

    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_valid_sig_dispatches_to_process(self, _mock_key, _mock_sender):
        client = _make_mock_client()
        # Set up a fake active project so _process_with_intent is called
        mock_project = MagicMock()
        mock_project.project_id = "proj_1"
        mock_project.root_path = "/tmp/repo"
        client._project_manager.get_project_for_chat.return_value = None
        client._project_manager.get_active_project.return_value = mock_project

        handler = _bootstrap(client)

        cmd = "/deep run"
        sig = _sign_command(cmd, "chat_4")
        handler("mid_4", "chat_4", None, {
            "command_text": cmd,
            "command_sig": sig,
        })

        client._reply_text.assert_not_called()
        client._process_with_intent.assert_called_once()
        # Verify the dispatched command matches
        args = client._process_with_intent.call_args[0]
        assert args[0] == "mid_4"
        assert args[1] == "chat_4"
        assert args[2] == cmd


class TestRetryCommandRepoLockProbe:
    """AC-24: Probe-acquire-then-release pattern on repo lock."""

    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_repo_lock_probe_acquire_then_release(self, _mock_key, _mock_sender):
        client = _make_mock_client()

        # Set up project with root_path
        mock_project = MagicMock()
        mock_project.project_id = "proj_1"
        mock_project.root_path = "/tmp/repo"
        client._project_manager.get_project_for_chat.return_value = mock_project

        # Set up repo lock manager on handler_ctx
        mock_repo_lock = MagicMock()
        mock_probe_result = MagicMock()
        mock_probe_result.success = True
        mock_repo_lock.acquire.return_value = mock_probe_result

        mock_ctx = MagicMock()
        mock_ctx.repo_lock_manager = mock_repo_lock
        client._handler_ctx = mock_ctx

        handler = _bootstrap(client)

        cmd = "/deep run"
        sig = _sign_command(cmd, "chat_5")
        handler("mid_5", "chat_5", "proj_1", {
            "command_text": cmd,
            "command_sig": sig,
        })

        # Verify acquire was called, then release, then _process_with_intent
        mock_repo_lock.acquire.assert_called_once_with(
            "/tmp/repo",
            "chat_5",
            owner_id=None,
        )
        mock_repo_lock.release.assert_called_once_with(
            "/tmp/repo",
            "chat_5",
            owner_id=None,
        )
        client._process_with_intent.assert_called_once()

    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_repo_lock_probe_fail_sends_conflict_card(self, _mock_key, _mock_sender):
        """When probe-acquire fails, a conflict card is sent instead of dispatch."""
        client = _make_mock_client()

        mock_project = MagicMock()
        mock_project.project_id = "proj_1"
        mock_project.root_path = "/tmp/repo"
        client._project_manager.get_project_for_chat.return_value = mock_project

        mock_repo_lock = MagicMock()
        mock_probe_result = MagicMock()
        mock_probe_result.success = False
        mock_probe_result.holder_chat_id = "other_chat"
        mock_probe_result.locked_since = 100.0
        mock_probe_result.last_active_time = 200.0
        mock_repo_lock.acquire.return_value = mock_probe_result

        mock_ctx = MagicMock()
        mock_ctx.repo_lock_manager = mock_repo_lock
        client._handler_ctx = mock_ctx

        # The adapter now delegates to client.send_lock_conflict_card (public method)
        client.send_lock_conflict_card = MagicMock()

        handler = _bootstrap(client)

        cmd = "/deep fix"
        sig = _sign_command(cmd, "chat_6")
        handler("mid_6", "chat_6", "proj_1", {
            "command_text": cmd,
            "command_sig": sig,
        })

        # Should NOT dispatch
        client._process_with_intent.assert_not_called()
        # Should NOT release (probe failed)
        mock_repo_lock.release.assert_not_called()
        # Should send conflict card
        client.send_lock_conflict_card.assert_called_once()


class TestRetryCommandTOCTOURace:
    """AC-25: Document the TOCTOU race between probe-release and re-acquire.

    This is a known design limitation: between the moment _handle_retry_command
    releases the probe lock and _process_with_intent re-acquires it, another
    chat could theoretically grab the lock.  This test documents this behavior.
    """

    @patch("src.thread.get_current_sender_id", return_value="sender_1")
    @patch("src.utils.signing._get_signing_key", return_value=_SIGNING_KEY)
    def test_toctou_race_between_probe_and_dispatch(self, _mock_key, _mock_sender):
        """Simulate another chat grabbing the lock after probe-release."""
        client = _make_mock_client()

        mock_project = MagicMock()
        mock_project.project_id = "proj_1"
        mock_project.root_path = "/tmp/repo"
        client._project_manager.get_project_for_chat.return_value = mock_project

        # Threading primitives for precise synchronization
        release_happened = threading.Event()
        race_acquired = threading.Event()

        mock_repo_lock = MagicMock()
        probe_result_ok = MagicMock(success=True)
        mock_repo_lock.acquire.return_value = probe_result_ok

        def release_side_effect(*args, **kwargs):
            """Signal that release happened, then wait briefly for racer."""
            release_happened.set()

        mock_repo_lock.release.side_effect = release_side_effect

        mock_ctx = MagicMock()
        mock_ctx.repo_lock_manager = mock_repo_lock
        client._handler_ctx = mock_ctx

        def racer():
            """Simulate another chat acquiring the lock after release."""
            if release_happened.wait(timeout=5):
                # At this point, the lock has been released by the probe
                # but _process_with_intent hasn't re-acquired it yet.
                # In production, another chat could acquire here.
                race_acquired.set()

        handler = _bootstrap(client)

        cmd = "/deep run"
        sig = _sign_command(cmd, "chat_7")

        racer_thread = threading.Thread(target=racer, daemon=True)
        racer_thread.start()

        handler("mid_7", "chat_7", "proj_1", {
            "command_text": cmd,
            "command_sig": sig,
        })

        racer_thread.join(timeout=5)

        # Document the race: the racer was able to act in the gap
        assert race_acquired.is_set(), (
            "TOCTOU window documented: another chat could act between "
            "probe-release and _process_with_intent re-acquire"
        )
        # Despite the race, _process_with_intent was still called
        # (it will independently re-acquire the lock)
        client._process_with_intent.assert_called_once()
