"""Activation guard for Slock passive auto-activation.

Controls who can trigger auto-activation and enforces rate limits
to prevent abuse and unintended activation storms.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.trust.models import EffectiveTrust

logger = logging.getLogger(__name__)

# Maximum number of unique users tracked for rate limiting
_MAX_USER_ENTRIES = 10000

# Activation denied reasons
ACTIVATION_DENIED_RATE_LIMIT = "rate_limit"
"""Denial reason: rate limit exceeded (per-user or global)."""

ACTIVATION_DENIED_ADMIN_REQUIRED = "admin_required"
"""Denial reason: whitelist is empty and policy is admin_only."""

ACTIVATION_DENIED_NOT_WHITELISTED = "not_whitelisted"
"""Denial reason: user is not in the whitelist."""

ACTIVATION_ALLOWED = "allowed"
"""Reason string when activation is permitted."""

ACTIVATION_DENIED_EXTERNAL = "external_or_unknown"


def _normalize_user_ids(value: object) -> frozenset[str]:
    """Normalize production id sets and legacy comma-separated test values."""
    if isinstance(value, str):
        return frozenset(uid.strip() for uid in value.split(",") if uid.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(
            uid.strip()
            for uid in value
            if isinstance(uid, str) and uid.strip()
        )
    return frozenset()


class ActivationGuard:
    """Guards auto-activation with permission checks and rate limiting.

    Permission model:
        - Admin users (from settings.admin_user_ids) can always activate
        - Users in whitelist (slock_auto_activate_whitelist_user_ids) can activate
        - With allow_all policy + passive mode: all users can activate
        - All others are denied

    Rate limiting (sliding window):
        - Per-user: max N activations per 60s window
        - Global: max M activations per 60s window
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        # Per-user timestamps: bounded LRU OrderedDict -> deque per user
        self._user_timestamps: OrderedDict[str, deque] = OrderedDict()
        # Global timestamps: bounded deque
        self._global_timestamps: deque = deque(maxlen=200)
        # Call counter for inline GC
        self._call_count: int = 0

    def can_auto_activate(
        self,
        sender_id: str,
        chat_id: str,
        settings: "Settings",
        *,
        effective_trust: "EffectiveTrust | None" = None,
    ) -> tuple[bool, str]:
        """Check if the sender is allowed to trigger auto-activation.

        Args:
            sender_id: The open_id of the message sender.
            chat_id: The chat_id where activation is requested.
            settings: Application settings instance.

        Returns:
            A tuple of (allowed, reason):
            - allowed: True if activation is permitted, False otherwise.
            - reason: A string indicating the reason. One of:
                - ACTIVATION_ALLOWED: activation permitted
                - ACTIVATION_DENIED_RATE_LIMIT: rate limit exceeded
                - ACTIVATION_DENIED_ADMIN_REQUIRED: admin-only policy
                - ACTIVATION_DENIED_NOT_WHITELISTED: not in whitelist
        """
        if effective_trust is not None:
            from src.trust.models import TrustZone

            if effective_trust.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP:
                return False, ACTIVATION_DENIED_EXTERNAL

        # Inline GC: every 50 calls trigger a full purge
        self._call_count += 1
        if self._call_count % 50 == 0:
            self.purge_stale()

        # Step 1: Permission check
        has_perm, reason = self._has_permission(sender_id, settings)
        if not has_perm:
            logger.debug(
                "Auto-activate denied for user=%s in chat=%s: %s",
                sender_id,
                chat_id,
                reason,
            )
            return False, reason

        # Step 2: Rate limit check
        if not self._check_rate_limit(sender_id, settings):
            logger.warning(
                "Auto-activate rate-limited for user=%s in chat=%s",
                sender_id,
                chat_id,
            )
            return False, ACTIVATION_DENIED_RATE_LIMIT

        return True, ACTIVATION_ALLOWED

    def _has_permission(
        self,
        sender_id: str,
        settings: "Settings",
    ) -> tuple[bool, str]:
        """Check if sender has permission to auto-activate.

        Returns:
            A tuple of (has_permission, reason):
            - has_permission: True if sender has permission, False otherwise.
            - reason: ACTIVATION_ALLOWED if allowed, or one of the denial reasons.
        """
        if not sender_id:
            return False, ACTIVATION_DENIED_NOT_WHITELISTED

        # Admin users always allowed
        admin_ids = _normalize_user_ids(getattr(settings, "admin_user_ids", frozenset()))
        if sender_id in admin_ids:
            return True, ACTIVATION_ALLOWED

        # Whitelist check
        whitelist_str = getattr(settings, "slock_auto_activate_whitelist_user_ids", "") or ""
        if whitelist_str:
            whitelist = {uid.strip() for uid in whitelist_str.split(",") if uid.strip()}
            if sender_id in whitelist:
                return True, ACTIVATION_ALLOWED

        # Default policy when no whitelist is configured
        if not whitelist_str:
            policy = getattr(settings, "slock_auto_activate_default_policy", "allow_all")
            if policy == "allow_all" and getattr(settings, "slock_passive_mode", True):
                return True, ACTIVATION_ALLOWED
            # admin_only: only admin (already checked above) can activate
            return False, ACTIVATION_DENIED_ADMIN_REQUIRED

        # Whitelist exists but user not in it
        return False, ACTIVATION_DENIED_NOT_WHITELISTED

    def _check_rate_limit(self, sender_id: str, settings: "Settings") -> bool:
        """Check and record activation against rate limits."""
        now = time.time()
        window = 60.0  # 1-minute sliding window

        per_user_limit = getattr(settings, "slock_auto_activate_rate_limit_per_user", 3)
        global_limit = getattr(settings, "slock_auto_activate_rate_limit_global", 10)

        with self._lock:
            # Clean expired global entries
            while self._global_timestamps and now - self._global_timestamps[0] >= window:
                self._global_timestamps.popleft()

            # Clean expired per-user entries
            if sender_id in self._user_timestamps:
                user_ts = self._user_timestamps[sender_id]
                while user_ts and now - user_ts[0] >= window:
                    user_ts.popleft()
                if not user_ts:
                    del self._user_timestamps[sender_id]

            # Check global limit
            if len(self._global_timestamps) >= global_limit:
                return False

            # Check per-user limit
            user_ts = self._user_timestamps.get(sender_id)
            if user_ts and len(user_ts) >= per_user_limit:
                return False

            # Record this activation
            self._global_timestamps.append(now)
            if sender_id not in self._user_timestamps:
                # Enforce bounded size: evict oldest entry if at capacity
                if len(self._user_timestamps) >= _MAX_USER_ENTRIES:
                    self._user_timestamps.popitem(last=False)
                self._user_timestamps[sender_id] = deque(maxlen=per_user_limit * 2)
            self._user_timestamps[sender_id].append(now)
            # Move to end for LRU ordering
            self._user_timestamps.move_to_end(sender_id)
            return True

    def reset(self) -> None:
        """Reset all rate limit counters (useful for testing)."""
        with self._lock:
            self._user_timestamps.clear()
            self._global_timestamps.clear()

    def purge_stale(self, window: float = 60.0) -> int:
        """Remove all expired entries from rate limit tracking.

        Intended to be called periodically (e.g., every 5 minutes) by an
        external timer to prevent unbounded memory growth from inactive users.

        Returns:
            Number of user keys removed.
        """
        now = time.time()
        removed = 0
        with self._lock:
            # Clean global timestamps
            while self._global_timestamps and now - self._global_timestamps[0] >= window:
                self._global_timestamps.popleft()

            stale_keys = [
                uid for uid, timestamps in self._user_timestamps.items()
                if not any(now - ts < window for ts in timestamps)
            ]
            for uid in stale_keys:
                del self._user_timestamps[uid]
                removed += 1
        return removed


# Module-level singleton instance
_guard = ActivationGuard()


def _start_purge_timer():
    """Start a daemon Timer that calls purge_stale() every 300 seconds."""
    def _run():
        try:
            _guard.purge_stale()
        except Exception:
            logger.debug("purge_stale() raised; timer chain continues", exc_info=True)
        finally:
            # Always reschedule to prevent chain breakage
            _start_purge_timer()

    t = threading.Timer(300.0, _run)
    t.daemon = True
    t.start()


_start_purge_timer()


def get_activation_guard() -> ActivationGuard:
    """Get the module-level ActivationGuard singleton."""
    return _guard
