"""Unit tests for src/utils/signing.py (HMAC command-signature utilities).

Covers both legacy v1 (HMAC-only) and new v2 (nonce+exp+chat_id) signing.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.utils.signing import (
    VerifyResult,
    _compute_command_sig,
    _get_signing_key,
    _is_v2_sig,
    _record_nonce,
    sign_command,
    verify_command_sig,
)

_TEST_KEY = "test_secret_signing_key"


@pytest.fixture(autouse=True)
def _isolated_nonce_store(tmp_path, monkeypatch):
    """Keep persistent anti-replay state isolated for every signing test."""
    import src.utils.signing as signing

    store_path = tmp_path / "used-command-nonces.json"
    monkeypatch.setattr(
        signing,
        "_nonce_store_path",
        lambda: store_path,
        raising=False,
    )
    signing._USED_NONCES.clear()
    yield
    signing._USED_NONCES.clear()


# ---------------------------------------------------------------------------
# _get_signing_key
# ---------------------------------------------------------------------------


class TestGetSigningKey:

    def test_returns_app_secret(self):
        mock_settings = MagicMock()
        mock_settings.app_secret = "my_secret"
        with patch("src.config.get_settings", return_value=mock_settings):
            assert _get_signing_key() == "my_secret"

    def test_fallback_empty_on_exception(self):
        with patch("src.config.get_settings", side_effect=RuntimeError("boom")):
            assert _get_signing_key() == ""


# ---------------------------------------------------------------------------
# _compute_command_sig (legacy v1)
# ---------------------------------------------------------------------------


class TestComputeCommandSig:

    def test_produces_hmac_sha256(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            sig = _compute_command_sig("hello")
        expected = hmac.new(
            _TEST_KEY.encode(), "hello".encode(), hashlib.sha256
        ).hexdigest()
        assert sig == expected

    def test_different_keys_different_sigs(self):
        with patch("src.utils.signing._get_signing_key", return_value="key_a"):
            sig_a = _compute_command_sig("hello")
        with patch("src.utils.signing._get_signing_key", return_value="key_b"):
            sig_b = _compute_command_sig("hello")
        assert sig_a != sig_b

    def test_raises_on_empty_key(self):
        with patch("src.utils.signing._get_signing_key", return_value=""):
            with pytest.raises(ValueError, match="signing key is empty"):
                _compute_command_sig("any")


# ---------------------------------------------------------------------------
# sign_command (v2: nonce + exp + chat_id)
# ---------------------------------------------------------------------------


class TestSignCommand:

    def test_returns_dot_separated_payload(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_abc123")
        parts = payload.split(".")
        assert len(parts) == 4

    def test_exp_is_future_timestamp(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_abc123", ttl_seconds=3600)
        exp_str = payload.split(".")[1]
        exp = int(exp_str)
        assert exp > time.time()
        assert exp <= time.time() + 3601

    def test_different_commands_different_sigs(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            s1 = sign_command("/deploy", "chat1")
            s2 = sign_command("/rollback", "chat1")
        assert s1.split(".")[0] != s2.split(".")[0]

    def test_different_chats_different_sigs(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            s1 = sign_command("/deploy", "chat_A")
            s2 = sign_command("/deploy", "chat_B")
        assert s1.split(".")[3] != s2.split(".")[3]

    def test_raises_on_empty_key(self):
        with patch("src.utils.signing._get_signing_key", return_value=""):
            with pytest.raises(ValueError, match="signing key is empty"):
                sign_command("/deploy", "chat1")


# ---------------------------------------------------------------------------
# _is_v2_sig
# ---------------------------------------------------------------------------


class TestIsV2Sig:

    def test_v2_format_detected(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/cmd", "chat1")
        assert _is_v2_sig(payload) is True

    def test_v1_format_not_v2(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            sig = _compute_command_sig("/cmd")
        assert _is_v2_sig(sig) is False

    def test_random_string_not_v2(self):
        assert _is_v2_sig("not_a_signature") is False

    def test_empty_not_v2(self):
        assert _is_v2_sig("") is False


# ---------------------------------------------------------------------------
# verify_command_sig — v2 format
# ---------------------------------------------------------------------------


class TestVerifyCommandSigV2:

    def test_valid_v2_sig_accepted(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_123")
            result = verify_command_sig("/deploy", payload, chat_id="chat_123")
        assert result is VerifyResult.OK
        assert bool(result) is True

    def test_expired_sig_rejected(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            # Sign with ttl=0 (already expired)
            payload = sign_command("/deploy", "chat_123", ttl_seconds=-1)
            result = verify_command_sig("/deploy", payload, chat_id="chat_123")
        assert result is VerifyResult.EXPIRED
        assert bool(result) is False

    def test_nonce_replay_rejected(self):
        """Second verification with same payload should fail (nonce reuse)."""
        import src.utils.signing as _mod
        # Clear nonce store to avoid interference from other tests
        _mod._USED_NONCES.clear()
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_123")
            # First verify should succeed
            result1 = verify_command_sig("/deploy", payload, chat_id="chat_123")
            assert result1 is VerifyResult.OK
            # Second verify should fail (nonce already consumed)
            result2 = verify_command_sig("/deploy", payload, chat_id="chat_123")
            assert result2 is VerifyResult.NONCE_REUSED
            assert bool(result2) is False

    def test_nonce_replay_rejected_after_process_memory_is_reset(self):
        """A service restart must not make a consumed action reusable."""
        import src.utils.signing as _mod

        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("./restart.sh rr", "chat_123")
            assert verify_command_sig(
                "./restart.sh rr",
                payload,
                chat_id="chat_123",
            ) is VerifyResult.OK

            # Simulate a fresh process: the in-memory acceleration cache is gone,
            # while the durable nonce database remains.
            _mod._USED_NONCES.clear()

            assert verify_command_sig(
                "./restart.sh rr",
                payload,
                chat_id="chat_123",
            ) is VerifyResult.NONCE_REUSED

    def test_nonce_store_failure_rejects_otherwise_valid_action(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Anti-replay persistence is fail-closed, never best-effort."""
        import src.utils.signing as _mod

        invalid_store = tmp_path / "store-is-a-directory"
        invalid_store.mkdir()
        monkeypatch.setattr(_mod, "_nonce_store_path", lambda: invalid_store)
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("./restart.sh rr", "chat_123")
            result = verify_command_sig(
                "./restart.sh rr",
                payload,
                chat_id="chat_123",
            )

        assert result is VerifyResult.MISMATCH

    def test_chat_id_mismatch_rejected(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_A")
            result = verify_command_sig("/deploy", payload, chat_id="chat_B")
        assert result is VerifyResult.CHAT_MISMATCH
        assert bool(result) is False

    def test_tampered_command_rejected(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_123")
            result = verify_command_sig("/tampered", payload, chat_id="chat_123")
        assert result is VerifyResult.MISMATCH
        assert bool(result) is False

    def test_tampered_sig_rejected(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_123")
        # Tamper with the sig portion
        parts = payload.split(".")
        parts[0] = "a" * 64
        tampered = ".".join(parts)
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            result = verify_command_sig("/deploy", tampered, chat_id="chat_123")
        assert result is VerifyResult.MISMATCH

    def test_v2_without_chat_id_still_verifies(self):
        """When chat_id is not provided, v2 sig is verified without chat check."""
        import src.utils.signing as _mod
        _mod._USED_NONCES.clear()
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/deploy", "chat_123")
            # Verify without chat_id — should still check nonce+exp+hmac
            result = verify_command_sig("/deploy", payload)
        assert result is VerifyResult.OK


# ---------------------------------------------------------------------------
# verify_command_sig — v1 format (backward compatibility)
# ---------------------------------------------------------------------------


class TestVerifyCommandSigV1Compat:

    def test_v1_sig_accepted(self):
        """Old HMAC-only format should still pass within compat window."""
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            sig = _compute_command_sig("cmd")
            result = verify_command_sig("cmd", sig)
            assert result is VerifyResult.OK
            assert bool(result) is True

    def test_v1_sig_rejected_when_callback_has_chat_id(self):
        """Unbound legacy actions must not survive into real chat callbacks."""
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            sig = _compute_command_sig("cmd")
            result = verify_command_sig("cmd", sig, chat_id="any_chat")
            assert result is VerifyResult.MISMATCH

    def test_v1_wrong_sig_rejected(self):
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            result = verify_command_sig("cmd", "deadbeef" * 8)
        assert result is VerifyResult.MISMATCH

    @pytest.mark.parametrize(
        ("deploy_date", "window_days"),
        [
            (date.today().isoformat(), 7),
            ("", 7),
            ("2000-01-01", 36500),
        ],
    )
    def test_plain_sha256_is_always_rejected(
        self,
        deploy_date: str,
        window_days: int,
    ):
        """Unkeyed signatures stay invalid across deploys and process restarts."""
        cmd = "/status"
        plain_sig = hashlib.sha256(cmd.encode()).hexdigest()
        stale_settings = MagicMock(
            app_secret=_TEST_KEY,
            sig_compat_deploy_date=deploy_date,
            sig_compat_window_days=window_days,
        )

        with patch("src.config.get_settings", return_value=stale_settings):
            assert verify_command_sig(cmd, plain_sig) is VerifyResult.MISMATCH

    def test_plain_sha256_compatibility_window_is_not_configurable(self):
        from src.config.settings import Settings

        assert "sig_compat_deploy_date" not in Settings.model_fields
        assert "sig_compat_window_days" not in Settings.model_fields


# ---------------------------------------------------------------------------
# _record_nonce
# ---------------------------------------------------------------------------


class TestRecordNonce:

    def test_first_use_returns_false(self):
        import src.utils.signing as _mod
        _mod._USED_NONCES.clear()
        assert _record_nonce("unique_nonce_1") is False

    def test_reuse_returns_true(self):
        import src.utils.signing as _mod
        _mod._USED_NONCES.clear()
        _record_nonce("reused_nonce")
        assert _record_nonce("reused_nonce") is True

    def test_fails_closed_instead_of_evicting_live_nonces(self, monkeypatch):
        import src.utils.signing as _mod
        _mod._USED_NONCES.clear()
        monkeypatch.setattr(_mod, "_MAX_NONCES", 3)
        for i in range(3):
            _record_nonce(f"nonce_{i}")
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            _record_nonce("new_nonce")
        assert len(_mod._USED_NONCES) == 3
        assert "nonce_0" in _mod._USED_NONCES
        assert "new_nonce" not in _mod._USED_NONCES


# ---------------------------------------------------------------------------
# VerifyResult enum
# ---------------------------------------------------------------------------


class TestVerifyResult:

    def test_ok_is_truthy(self):
        assert bool(VerifyResult.OK) is True

    def test_mismatch_is_falsy(self):
        assert bool(VerifyResult.MISMATCH) is False


# ---------------------------------------------------------------------------
# Integration: full round-trip sign → verify
# ---------------------------------------------------------------------------


class TestSignVerifyRoundTrip:

    def test_sign_then_verify_same_chat(self):
        """Full round-trip: sign and verify with same command and chat_id."""
        import src.utils.signing as _mod
        _mod._USED_NONCES.clear()
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/restart", "oc_chatid_xyz")
            result = verify_command_sig("/restart", payload, chat_id="oc_chatid_xyz")
        assert result is VerifyResult.OK

    def test_sign_then_verify_wrong_chat(self):
        """Signature bound to one chat cannot be used in another."""
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/restart", "chat_A")
            result = verify_command_sig("/restart", payload, chat_id="chat_B")
        assert result is VerifyResult.CHAT_MISMATCH

    def test_expired_sig_roundtrip(self):
        """Signature with negative TTL is expired immediately."""
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/restart", "chat_A", ttl_seconds=-10)
            result = verify_command_sig("/restart", payload, chat_id="chat_A")
        assert result is VerifyResult.EXPIRED

    def test_replay_protection_roundtrip(self):
        """Same payload cannot be verified twice (nonce replay)."""
        import src.utils.signing as _mod
        _mod._USED_NONCES.clear()
        with patch("src.utils.signing._get_signing_key", return_value=_TEST_KEY):
            payload = sign_command("/restart", "chat_A")
            r1 = verify_command_sig("/restart", payload, chat_id="chat_A")
            r2 = verify_command_sig("/restart", payload, chat_id="chat_A")
        assert r1 is VerifyResult.OK
        assert r2 is VerifyResult.NONCE_REUSED
