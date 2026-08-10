"""HMAC command-signature utilities.

Provides tamper-detection signing and verification for Feishu card action
payloads.  Extracted from ``card.builders.lock`` to keep cryptographic
logic separate from UI card construction (Single Responsibility Principle).

v2 adds nonce + expiry + chat_id binding to prevent replay attacks.
The old keyed HMAC-only format remains supported for callers without chat context.
"""

from __future__ import annotations

import enum
import fcntl
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

# Same-process nonce cache; durable state is maintained by _record_nonce().
_USED_NONCES: OrderedDict[str, float] = OrderedDict()
_MAX_NONCES = 10000
_NONCE_STORE_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


class VerifyResult(enum.Enum):
    """Outcome of :func:`verify_command_sig`.

    * ``OK`` — signature valid (truthy).
    * ``MISMATCH`` — signature invalid, no special context (falsy).
    * ``EXPIRED`` — new-format signature has expired (falsy).
    * ``NONCE_REUSED`` — nonce was already consumed (replay attempt, falsy).
    * ``CHAT_MISMATCH`` — chat_id does not match the signed chat (falsy).
    """

    OK = "ok"
    MISMATCH = "mismatch"
    EXPIRED = "expired"
    NONCE_REUSED = "nonce_reused"
    CHAT_MISMATCH = "chat_mismatch"

    def __bool__(self) -> bool:  # noqa: D105
        return self is VerifyResult.OK


def _get_signing_key() -> str:
    """Return the HMAC signing key from settings.

    Falls back to an empty string only when settings are unavailable
    (e.g. during import-time or early bootstrap).  At runtime the key
    is ``settings.app_secret`` which is rejected by
    ``validate_feishu_config()`` at startup — process exits if empty.
    """
    try:
        from src.config import get_settings
        return get_settings().app_secret
    except Exception:
        logger.warning("Failed to retrieve signing key, falling back to empty string", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Legacy (v1) signing — retained for backward compatibility
# ---------------------------------------------------------------------------


def _compute_command_sig(command_text: str) -> str:
    """Return an HMAC-SHA256 hex digest of *command_text* for tamper detection.

    Uses ``settings.app_secret`` as the HMAC key so that external
    parties cannot forge a valid signature even though *command_text*
    is visible in the card payload.

    .. deprecated::
        Use :func:`sign_command` for new code. This function is retained
        for keyed legacy callers that genuinely lack chat context.
    """
    key = _get_signing_key()
    if not key:
        raise ValueError("signing key is empty")
    return hmac.new(key.encode("utf-8"), command_text.encode("utf-8"), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# New (v2) signing with nonce + expiry + chat_id binding
# ---------------------------------------------------------------------------


def sign_command(command_text: str, chat_id: str, ttl_seconds: int = 3600) -> str:
    """Sign command with nonce, expiry, and chat_id binding.

    Returns a dot-separated payload: ``sig.exp.nonce.chat_hash``

    Parameters
    ----------
    command_text : str
        The command text to sign.
    chat_id : str
        The chat_id to bind the signature to.
    ttl_seconds : int
        Time-to-live in seconds (default 1 hour).

    Returns
    -------
    str
        Dot-separated payload containing sig, expiry, nonce, and chat_id hash.
    """
    key = _get_signing_key()
    if not key:
        raise ValueError("signing key is empty")

    nonce = secrets.token_urlsafe(16)
    exp = int(time.time()) + ttl_seconds
    chat_hash = hashlib.sha256(chat_id.encode()).hexdigest()[:8]
    message = f"{command_text}|{chat_hash}|{exp}|{nonce}"
    sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    # Pack as: sig.exp.nonce.chat_hash
    return f"{sig}.{exp}.{nonce}.{chat_hash}"


def _is_v2_sig(sig_payload: str) -> bool:
    """Return True if the signature payload is in v2 dot-separated format."""
    parts = sig_payload.split(".")
    # v2 format: sig(64 hex).exp(digits).nonce(url-safe base64).chat_hash(8 hex)
    return len(parts) == 4 and len(parts[0]) == 64 and parts[1].isdigit()


def _nonce_store_path() -> Path:
    """Return the durable command-action nonce store path."""
    return Path.home() / ".ghostap" / "used-command-nonces.json"


def _record_nonce(nonce: str, expires_at: int | None = None) -> bool:
    """Record a nonce and return True if it was already seen (replay).

    A flock-protected JSON file is the authority so a service restart cannot
    make an old action reusable. The in-memory map is only a fast path.
    """
    now = int(time.time())
    expiry = int(expires_at if expires_at is not None else now + 3600)
    with _NONCE_STORE_LOCK:
        cached_expiry = _USED_NONCES.get(nonce)
        if cached_expiry is not None:
            if cached_expiry >= now:
                return True
            _USED_NONCES.pop(nonce, None)

        store_path = _nonce_store_path()
        store_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = Path(f"{store_path}.lock")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        lock_fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | nofollow | cloexec,
            0o600,
        )
        temp_path: str | None = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            records: dict[str, int] = {}
            try:
                store_fd = os.open(
                    store_path,
                    os.O_RDONLY | nofollow | cloexec,
                )
            except FileNotFoundError:
                store_fd = -1
            if store_fd >= 0:
                try:
                    stat_result = os.fstat(store_fd)
                    if stat_result.st_size > 2 * 1024 * 1024:
                        raise RuntimeError("command nonce store is oversized")
                    payload = bytearray()
                    while len(payload) <= 2 * 1024 * 1024:
                        chunk = os.read(store_fd, 64 * 1024)
                        if not chunk:
                            break
                        payload.extend(chunk)
                    raw_records = json.loads(payload.decode("utf-8")) if payload else {}
                    if not isinstance(raw_records, dict):
                        raise RuntimeError("command nonce store has invalid shape")
                    for stored_nonce, stored_expiry in raw_records.items():
                        if (
                            not isinstance(stored_nonce, str)
                            or isinstance(stored_expiry, bool)
                            or not isinstance(stored_expiry, (int, float))
                        ):
                            raise RuntimeError("command nonce store has invalid entry")
                        if int(stored_expiry) >= now:
                            records[stored_nonce] = int(stored_expiry)
                finally:
                    os.close(store_fd)

            stored_expiry = records.get(nonce)
            if stored_expiry is not None:
                _USED_NONCES[nonce] = float(stored_expiry)
                return True

            if len(records) >= _MAX_NONCES:
                raise RuntimeError("command nonce store capacity exhausted")

            records[nonce] = expiry
            temp_fd, temp_path = tempfile.mkstemp(
                prefix=".used-command-nonces.",
                dir=store_path.parent,
            )
            try:
                os.fchmod(temp_fd, 0o600)
                with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                    json.dump(
                        records,
                        temp_file,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(temp_path, store_path)
                temp_path = None
                directory_fd = os.open(
                    store_path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cloexec,
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                raise
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

        expired_cached = [
            cached_nonce
            for cached_nonce, cached_until in _USED_NONCES.items()
            if cached_until < now
        ]
        for cached_nonce in expired_cached:
            _USED_NONCES.pop(cached_nonce, None)
        _USED_NONCES[nonce] = float(expiry)
        return False


def _verify_v2_sig(command_text: str, chat_id: str, sig_payload: str) -> VerifyResult:
    """Verify a v2 signature payload.

    Checks:
    1. Structural validity (4 dot-separated parts)
    2. Expiry (exp >= now)
    3. Nonce not reused (anti-replay)
    4. HMAC recomputation
    5. chat_id hash match
    """
    parts = sig_payload.split(".")
    if len(parts) != 4:
        return VerifyResult.MISMATCH

    sig_hex, exp_str, nonce, payload_chat_hash = parts

    # Validate expiry
    try:
        exp = int(exp_str)
    except ValueError:
        return VerifyResult.MISMATCH

    if time.time() > exp:
        return VerifyResult.EXPIRED

    # Check chat_id hash matches
    chat_hash = hashlib.sha256(chat_id.encode()).hexdigest()[:8]
    if not hmac.compare_digest(payload_chat_hash, chat_hash):
        return VerifyResult.CHAT_MISMATCH

    # Recompute HMAC
    key = _get_signing_key()
    if not key:
        return VerifyResult.MISMATCH

    message = f"{command_text}|{payload_chat_hash}|{exp_str}|{nonce}"
    expected_sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(sig_hex, expected_sig):
        return VerifyResult.MISMATCH

    # Check nonce not reused (do this AFTER sig verification to avoid
    # poisoning the nonce store with forged payloads)
    try:
        if _record_nonce(nonce, exp):
            return VerifyResult.NONCE_REUSED
    except Exception:
        logger.error("Command nonce store failed closed", exc_info=True)
        return VerifyResult.MISMATCH

    return VerifyResult.OK


# ---------------------------------------------------------------------------
# Unified verification interface
# ---------------------------------------------------------------------------


def verify_command_sig(command_text: str, sig: str, *, chat_id: str = "") -> VerifyResult:
    """Verify *sig* against the command text, with optional chat_id binding.

    Supports v2 (nonce+exp+chat_id) and keyed v1 (HMAC-only) formats:

    1. If *sig* is in v2 format and *chat_id* is provided, verify as v2.
    2. If a callback supplies *chat_id*, reject every unbound legacy format.
    3. Otherwise accept a matching keyed v1 HMAC-SHA256 signature.
    4. Reject every unkeyed digest, including legacy plain SHA-256.

    Returns :attr:`VerifyResult.OK` when valid.  ``VerifyResult`` implements
    ``__bool__`` so that ``if verify_command_sig(...)`` continues to work
    as a truthy/falsy check for backward compatibility.
    """
    if not sig:
        return VerifyResult.MISMATCH

    # Try v2 format first
    if _is_v2_sig(sig):
        if chat_id:
            return _verify_v2_sig(command_text, chat_id, sig)
        # v2 sig but no chat_id provided — extract chat_hash from payload
        # and skip chat_id check (caller doesn't have it)
        # This shouldn't normally happen but handles edge cases gracefully
        parts = sig.split(".")
        if len(parts) == 4:
            # Verify without chat_id constraint (still checks nonce+exp+hmac)
            return _verify_v2_sig_without_chat_check(command_text, sig)
        return VerifyResult.MISMATCH

    # A real card callback always carries its chat id. Reject unbound legacy
    # signatures there so old destructive buttons cannot be copied across
    # chats or replayed indefinitely. Legacy verification remains available
    # only to callers that genuinely lack chat context during migration.
    if chat_id:
        return VerifyResult.MISMATCH

    # Try v1 HMAC-SHA256 format
    try:
        expected_hmac = _compute_command_sig(command_text)
        if hmac.compare_digest(sig, expected_hmac):
            return VerifyResult.OK
    except ValueError:
        pass  # empty key — skip HMAC check

    return VerifyResult.MISMATCH


def _verify_v2_sig_without_chat_check(command_text: str, sig_payload: str) -> VerifyResult:
    """Verify a v2 signature without enforcing chat_id match.

    Used when the verifier does not have access to chat_id (edge case).
    Still checks expiry, nonce, and HMAC.
    """
    parts = sig_payload.split(".")
    if len(parts) != 4:
        return VerifyResult.MISMATCH

    sig_hex, exp_str, nonce, payload_chat_hash = parts

    # Validate expiry
    try:
        exp = int(exp_str)
    except ValueError:
        return VerifyResult.MISMATCH

    if time.time() > exp:
        return VerifyResult.EXPIRED

    # Recompute HMAC using the chat_hash from the payload itself
    key = _get_signing_key()
    if not key:
        return VerifyResult.MISMATCH

    message = f"{command_text}|{payload_chat_hash}|{exp_str}|{nonce}"
    expected_sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(sig_hex, expected_sig):
        return VerifyResult.MISMATCH

    # Check nonce not reused
    try:
        if _record_nonce(nonce, exp):
            return VerifyResult.NONCE_REUSED
    except Exception:
        logger.error("Command nonce store failed closed", exc_info=True)
        return VerifyResult.MISMATCH

    return VerifyResult.OK

