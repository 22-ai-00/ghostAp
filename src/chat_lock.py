"""Chat-level lock manager (in-memory, single-process).

Allows administrators to lock a chat so that non-admin users' messages
are rejected (within GhostAP's message-processing scope).

已知限制（V1）：
- 纯内存单进程实现，进程重启后所有群锁状态丢失。
- 多实例部署场景下锁不互通，不适用于分布式部署。
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from src.config import get_settings
from src.utils.env import is_test_environment
from src.utils.lock_order import LockLevel, ordered_lock


class ChatLockCode(str, enum.Enum):
    """Structured result codes for chat lock/unlock operations.

    Each member's *value* matches the corresponding key in ``UI_TEXT``
    so that the presentation layer can resolve it via
    ``UI_TEXT[code.value].format(**format_params)``.

    Members that require ``.format()`` parameters:
    - ``CONTACT_NAMED_UNLOCK`` — requires ``name`` (locker display name).
    """

    NO_ADMIN_CONFIG = "chat_lock_no_admin_config"
    NO_ADMIN_CONFIG_USER = "chat_lock_no_admin_config_user"
    CONTACT_ADMIN_TO_LOCK = "chat_lock_contact_admin_to_lock"
    ALREADY_LOCKED = "chat_lock_already_locked"
    LOCKED_SUCCESS = "chat_lock_locked_success"
    CONTACT_NAMED_UNLOCK = "chat_lock_contact_named_unlock"
    CONTACT_ADMIN_UNLOCK = "chat_lock_contact_admin_unlock"
    NOT_LOCKED = "chat_lock_not_locked"
    UNLOCKED_SUCCESS = "chat_lock_unlocked_success"

logger = logging.getLogger(__name__)

# Commands that are always allowed even when a chat is locked.
# Add new read-only commands here when they are introduced.
READONLY_COMMANDS: frozenset[str] = frozenset({
    "/status", "/help", "/帮助", "/menu", "/projects",
    "/lock", "/unlock", "/setadmin", "/exit", "/quit",
})

# Safe interrupt commands — allowed during lock so users can abort running tasks.
# Separated from READONLY_COMMANDS because they mutate state (stop engines),
# but are considered safe even during lock-down.
SAFE_INTERRUPT_COMMANDS: frozenset[str] = frozenset({
    "/stop_deep", "/stop_spec",
})


@dataclass
class ChatLockEntry:
    """Internal bookkeeping for a locked chat."""

    locked_by: str  # user_id of admin who locked
    locked_at: float  # monotonic timestamp
    locked_at_wall: float = field(default_factory=time.time)  # wall-clock for display
    locked_by_name: str = ""  # display name (best-effort, may be empty)


@dataclass(frozen=True)
class ChatLockInfo:
    """Immutable snapshot of a chat lock — returned by :meth:`ChatLockManager.get_lock_info`.

    Field names mirror :class:`ChatLockEntry` for drop-in compatibility,
    but the instance is frozen to prevent callers from mutating internal state.
    """

    locked_by: str
    locked_at: float
    locked_at_wall: float
    locked_by_name: str = ""


@dataclass
class ChatLockResult:
    """Outcome of a lock/unlock attempt.

    The *code* field carries a structured :class:`ChatLockCode` enum value.
    Callers in the presentation layer resolve it to a UI string via
    ``UI_TEXT[result.code.value].format(**result.format_params)``.
    """

    success: bool
    code: Optional[ChatLockCode] = None
    format_params: dict = field(default_factory=dict)
    idempotent: bool = False


class ChatLockManager:
    """Singleton chat-level lock manager (thread-safe, in-memory).

    When a chat is locked:
    - Only admin users may send messages processed by GhostAP.
    - Non-admin messages are rejected with a friendly prompt.
    - Only admin users may modify GhostAP-managed chat settings
      (project binding, mode switching, etc.).
    """

    # Card actions that are always exempt from chat-lock blocking.
    # Keep these as local string literals to avoid a core-layer dependency on
    # ``src.card``.  tests/test_action_dispatch_mapping.py compares this set
    # with the canonical card action IDs to prevent drift.
    CARD_EXEMPT_ACTIONS: frozenset[str] = frozenset({
        "force_release_repo_lock",
        "confirm_lock",
        "cancel_lock",
        "confirm_force_release",
        "cancel_force_release",
        "help_category",
        "retry_command",
    })

    def __init__(
        self,
        *,
        max_duration: Optional[float] = None,
        cleanup_interval: Optional[float] = None,
        on_auto_unlock: Optional[Callable[[str, ChatLockEntry], None]] = None,
    ):
        self._locks: dict[str, ChatLockEntry] = {}
        self._mu = ordered_lock(LockLevel.CHAT_LOCK_MGR, name="ChatLockManager._mu")

        # TTL configuration
        try:
            settings = get_settings()
            self._max_duration: float = max_duration if max_duration is not None else settings.chat_lock_max_duration
            self._cleanup_interval: float = cleanup_interval if cleanup_interval is not None else settings.chat_lock_cleanup_interval
        except Exception:
            self._max_duration = max_duration if max_duration is not None else 86400
            self._cleanup_interval = cleanup_interval if cleanup_interval is not None else 60

        self._on_auto_unlock = on_auto_unlock

        # Daemon cleanup thread (lazy-started on first lock)
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._thread_mu = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Empty admins are a supported bootstrap state; /setadmin can initialize
        # the first admin without requiring a manual .env edit.
        try:
            settings = get_settings()
            if not settings.admin_user_ids:
                logger.info(
                    "ChatLockManager: admin_user_ids is empty — /lock and /unlock "
                    "will be unavailable until /setadmin initializes the bot admin."
                )
        except Exception:
            logger.debug("Failed to check admin_user_ids during init", exc_info=True)

    def is_admin(self, user_id: str) -> bool:
        """Check if *user_id* is in the configured admin list."""
        settings = get_settings()
        return user_id in settings.admin_user_ids

    @staticmethod
    def get_allowed_commands_display() -> str:
        """Return a formatted string of commands allowed during chat lock.

        Excludes admin-only commands (/lock, /unlock) and returns a
        space-separated backtick-quoted list suitable for embedding in
        Markdown cards.  This is the **single source of truth** for the
        presentation layer — card builders should call this method instead
        of importing READONLY_COMMANDS / SAFE_INTERRUPT_COMMANDS directly.
        """
        _admin_only = frozenset({"/lock", "/unlock", "/setadmin"})
        _cmds = sorted((READONLY_COMMANDS | SAFE_INTERRUPT_COMMANDS) - _admin_only)
        return " ".join(f"`{c}`" for c in _cmds) if _cmds else "`/help` `/status`"

    def lock_chat(self, chat_id: str, user_id: str, *, sender_name: str = "") -> ChatLockResult:
        """Lock a chat. Only admins may lock."""
        if not self.is_admin(user_id):
            settings = get_settings()
            if not settings.admin_user_ids:
                return ChatLockResult(success=False, code=ChatLockCode.NO_ADMIN_CONFIG_USER)
            return ChatLockResult(success=False, code=ChatLockCode.CONTACT_ADMIN_TO_LOCK)

        with self._mu:
            if chat_id in self._locks:
                return ChatLockResult(success=True, code=ChatLockCode.ALREADY_LOCKED, idempotent=True)
            self._locks[chat_id] = ChatLockEntry(
                locked_by=user_id,
                locked_at=time.monotonic(),
                locked_at_wall=time.time(),
                locked_by_name=sender_name,
            )
            logger.info("ChatLock: chat=%s locked by user=%s", chat_id[:12], user_id[:12])
            self._ensure_cleanup_thread()
            return ChatLockResult(success=True, code=ChatLockCode.LOCKED_SUCCESS)

    def unlock_chat(self, chat_id: str, user_id: str) -> ChatLockResult:
        """Unlock a chat. Only admins may unlock."""
        if not self.is_admin(user_id):
            settings = get_settings()
            if not settings.admin_user_ids:
                return ChatLockResult(success=False, code=ChatLockCode.NO_ADMIN_CONFIG_USER)
            # Include locker name in the hint when available
            with self._mu:
                entry = self._locks.get(chat_id)
                locker_name = entry.locked_by_name if entry and entry.locked_by_name else ""
            if locker_name:
                return ChatLockResult(
                    success=False,
                    code=ChatLockCode.CONTACT_NAMED_UNLOCK,
                    format_params={"name": locker_name},
                )
            return ChatLockResult(success=False, code=ChatLockCode.CONTACT_ADMIN_UNLOCK)

        with self._mu:
            entry = self._locks.pop(chat_id, None)
            if entry is None:
                return ChatLockResult(success=True, code=ChatLockCode.NOT_LOCKED, idempotent=True)
            logger.info("ChatLock: chat=%s unlocked by user=%s", chat_id[:12], user_id[:12])
            return ChatLockResult(success=True, code=ChatLockCode.UNLOCKED_SUCCESS)

    def is_locked(self, chat_id: str) -> bool:
        """Check if a chat is currently locked."""
        with self._mu:
            return chat_id in self._locks

    @runtime_checkable
    class _CommandMatchLike(Protocol):
        """最小 slash DTO 契约（避免 core 反向依赖 feishu 包）。

        下游只依赖：
        - command: canonical 命令（lowercase, e.g. "/deep"）
        - has_args: 是否携带参数（args 非空）
        """

        command: str
        has_args: bool

    def should_block(
        self,
        chat_id: str,
        user_id: str,
        *,
        command_match: Optional["ChatLockManager._CommandMatchLike"] = None,
    ) -> bool:
        """Return True if the message from *user_id* in *chat_id* should be blocked.

        - Read-only commands (``/status``, ``/help``, etc.) are always allowed.
        - If chat is not locked → False (allow).
        - If chat is locked and user is admin → False (allow).
        - If chat is locked and user is NOT admin → True (block).
        """
        cmd = (getattr(command_match, "command", "") or "").strip().lower() if command_match else ""

        # Read-only and safe-interrupt commands are never blocked.
        if cmd and (cmd in READONLY_COMMANDS or cmd in SAFE_INTERRUPT_COMMANDS):
            return False

        # Compute admin status outside the lock to avoid I/O inside the
        # critical section.  Note: get_settings() is a cached singleton
        # (double-checked locking in config.py) — no file/env I/O on the
        # hot path after first call.
        settings = get_settings()
        _is_admin = user_id in settings.admin_user_ids

        with self._mu:
            if chat_id not in self._locks:
                return False
            # Chat is locked — only admins may proceed.
            return not _is_admin

    def should_block_card_action(
        self, chat_id: str, user_id: str, action_type: str,
    ) -> bool:
        """Return True if a card action should be blocked by chat lock.

        Encapsulates all card-action exemption logic:
        - Not locked → False
        - Admin → False
        - Read-only / safe-interrupt commands → False
        - ``*_stop`` suffix (engine stop buttons) → False
        - ``show_*`` prefix (read-only views) → False
        - Action in :attr:`CARD_EXEMPT_ACTIONS` → False
        - Otherwise → True (block)
        """
        if not self.is_locked(chat_id):
            return False

        if self.is_admin(user_id):
            return False

        # Read-only and safe-interrupt commands are never blocked.
        if action_type in READONLY_COMMANDS or action_type in SAFE_INTERRUPT_COMMANDS:
            return False

        # Suffix / prefix patterns
        if action_type.endswith("_stop") or action_type.startswith("show_"):
            return False

        if action_type in self.CARD_EXEMPT_ACTIONS:
            return False

        return True

    def get_lock_info(self, chat_id: str) -> Optional[ChatLockInfo]:
        """Return an immutable snapshot of the lock entry, or None if unlocked."""
        with self._mu:
            entry = self._locks.get(chat_id)
            if entry is None:
                return None
            return ChatLockInfo(
                locked_by=entry.locked_by,
                locked_at=entry.locked_at,
                locked_at_wall=entry.locked_at_wall,
                locked_by_name=entry.locked_by_name,
            )

    # ------------------------------------------------------------------
    # TTL cleanup daemon
    # ------------------------------------------------------------------

    def _ensure_cleanup_thread(self) -> None:
        """Start the background cleanup thread if not already running."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        with self._thread_mu:
            if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
                return
            self._stop_event.clear()
            t = threading.Thread(target=self._cleanup_loop, name="ChatLockCleanup", daemon=True)
            self._cleanup_thread = t
            t.start()

    def _cleanup_loop(self) -> None:
        """Daemon loop: periodically check for expired chat locks."""
        while not self._stop_event.wait(timeout=self._cleanup_interval):
            try:
                self._cleanup_expired()
            except Exception:
                logger.exception("ChatLockCleanup: unexpected error in cleanup cycle")

    def _cleanup_expired(self) -> None:
        """Remove chat locks that have exceeded max_duration."""
        now = time.monotonic()
        expired: list[tuple[str, ChatLockEntry]] = []
        with self._mu:
            for chat_id, entry in list(self._locks.items()):
                if (now - entry.locked_at) >= self._max_duration:
                    expired.append((chat_id, entry))
                    del self._locks[chat_id]
                    logger.info(
                        "ChatLockCleanup: auto-unlocked chat=%s (locked for %.0fs, max=%ds)",
                        chat_id[:12], now - entry.locked_at, int(self._max_duration),
                    )
        # Fire callbacks outside the lock to avoid re-entrant acquisition
        if expired and self._on_auto_unlock:
            for chat_id, entry in expired:
                try:
                    self._on_auto_unlock(chat_id, entry)
                except Exception:
                    logger.exception("ChatLockCleanup: on_auto_unlock callback error for chat=%s", chat_id[:12])

    def shutdown(self) -> None:
        """Stop the cleanup thread gracefully."""
        self._stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[ChatLockManager] = None
_instance_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def get_chat_lock_manager() -> ChatLockManager:
    """Return the global ChatLockManager singleton (lazy-initialized)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ChatLockManager()
    return _instance


def set_chat_lock_manager(
    manager: ChatLockManager,
    *,
    is_test_env_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Set the global ChatLockManager singleton for dependency injection/testing."""
    check_fn = is_test_env_check if is_test_env_check is not None else is_test_environment
    if not check_fn():
        raise RuntimeError(
            "set_chat_lock_manager() is only allowed in test environments. "
            "Modifying global singletons in production can cause race conditions."
        )
    global _instance
    with _instance_lock:
        if _instance is not None and _instance is not manager:
            _instance.shutdown()
        _instance = manager


def shutdown_if_active() -> None:
    """Best-effort shutdown of the singleton if it was ever created.

    Safe to call multiple times (idempotent) and even before the singleton
    is initialized — in that case it simply does nothing.
    """
    if _instance is not None:
        try:
            _instance.shutdown()
        except Exception:
            logger.debug("ChatLockManager shutdown error", exc_info=True)


def _reset_chat_lock_manager_for_testing(
    *,
    is_test_env_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Reset the singleton. **Test-only.**"""
    check_fn = is_test_env_check if is_test_env_check is not None else is_test_environment
    if not check_fn():
        raise RuntimeError(
            "_reset_chat_lock_manager_for_testing() is only allowed in test environments. "
            "Modifying global singletons in production can cause race conditions."
        )
    global _instance
    with _instance_lock:
        if _instance is not None:
            _instance.shutdown()
        _instance = None
