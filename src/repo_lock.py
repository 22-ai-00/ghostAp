"""Repo-level mutex lock manager (in-memory, single-process).

Prevents concurrent requests from operating on the same git repository.
Supports request-owner reentrant acquire, p2p privilege bypass, and renewable
lease cleanup via a background daemon thread.

已知限制（V1）：
- 纯内存单进程实现，进程重启后所有仓库锁状态丢失（重启后所有子进程已终止，
  不存在活跃操作，因此丢失锁状态是安全的）。
- 多实例部署场景下锁不互通，不适用于分布式部署。
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.config import get_settings
from src.utils.lock_order import LockLevel, ordered_lock
from src.utils.path import normalize_repo_path

logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
_HEARTBEAT_RENEWALS_PER_LEASE = 3.0


def get_repo_lock_heartbeat_interval(manager: object) -> float:
    """Return a safe numeric heartbeat interval from a repo-lock manager."""

    raw = getattr(
        manager,
        "heartbeat_interval",
        _DEFAULT_HEARTBEAT_INTERVAL_S,
    )
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(raw)
        or raw <= 0
    ):
        return _DEFAULT_HEARTBEAT_INTERVAL_S
    return float(raw)


# ---------------------------------------------------------------------------
# Lightweight event emitter (Observer pattern)
# ---------------------------------------------------------------------------

class SimpleEvent:
    """Thread-safe multi-subscriber event emitter.

    Usage::

        event = SimpleEvent()
        unsub = event.subscribe(my_handler)
        event.emit("arg1", "arg2")  # calls my_handler("arg1", "arg2")
        unsub()  # removes my_handler
    """

    def __init__(self) -> None:
        self._handlers: list[Callable] = []
        self._mu = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def subscribe(self, handler: Callable) -> Callable[[], None]:
        """Add *handler*; return an unsubscribe callable."""
        with self._mu:
            self._handlers.append(handler)

        def _unsub() -> None:
            with self._mu:
                try:
                    self._handlers.remove(handler)
                except ValueError:
                    logger.debug("handler not found in list", exc_info=True)

        return _unsub

    def emit(self, *args, **kwargs) -> None:
        """Call all handlers with the given arguments (fire-and-forget)."""
        with self._mu:
            snapshot = list(self._handlers)
        for fn in snapshot:
            try:
                fn(*args, **kwargs)
            except Exception:
                logger.error("SimpleEvent callback failed", exc_info=True)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class LockConflictError(Exception):
    """Raised when a repo lock cannot be acquired due to another chat holding it."""

    def __init__(
        self,
        message: str = "",
        *,
        holder_chat_id: str = "",
        locked_since: float = 0.0,
        root_path: str = "",
        last_active_time: float = 0.0,
    ):
        super().__init__(message)
        self.holder_chat_id = holder_chat_id
        self.locked_since = locked_since
        self.root_path = root_path
        self.last_active_time = last_active_time


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RepoLockEntry:
    """Internal bookkeeping for a single repo lock."""

    chat_id: str
    owner_id: str = ""
    refcount: int = 1
    acquired_at: float = field(default_factory=time.monotonic)
    last_active_time: float = field(default_factory=time.monotonic)
    last_sender_id: str = ""


@dataclass
class AcquireResult:
    """Outcome of an acquire attempt."""

    success: bool
    holder_chat_id: Optional[str] = None
    locked_since: Optional[float] = None  # monotonic timestamp
    last_active_time: Optional[float] = None  # monotonic timestamp


@dataclass(frozen=True)
class RepoLockInfo:
    """Public-facing snapshot of one lock (for diagnostics)."""

    root_path: str
    chat_id: str
    refcount: int
    acquired_at: float
    last_active_time: float
    idle_seconds: float
    last_sender_id: str = ""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class RepoLockManager:
    """Singleton repo-level lock manager (thread-safe, in-memory)."""

    def __init__(
        self,
        *,
        idle_timeout: Optional[int] = None,
        cleanup_interval: Optional[int] = None,
        hard_timeout: Optional[int] = None,
        on_hard_timeout_reclaim: Optional[Callable[[str, str], None]] = None,
    ):
        settings = get_settings()
        self._idle_timeout = idle_timeout if idle_timeout is not None else settings.repo_lock_idle_timeout
        self._cleanup_interval = cleanup_interval if cleanup_interval is not None else settings.repo_lock_cleanup_interval
        self._hard_timeout = hard_timeout if hard_timeout is not None else settings.repo_lock_hard_timeout
        self.on_reclaim = SimpleEvent()
        if on_hard_timeout_reclaim is not None:
            self.on_reclaim.subscribe(on_hard_timeout_reclaim)
        self.on_release = SimpleEvent()
        self._locks: dict[str, RepoLockEntry] = {}
        self._blocked_chats: dict[str, set[str]] = {}  # normalized_path → set of blocked chat_ids (max 10)
        self._mu = ordered_lock(LockLevel.REPO_LOCK, name="RepoLockManager._mu")

        # Opaque token ↔ normalized path mapping (for card actions).
        # Prevents leaking full filesystem paths in Feishu card JSON.
        self._token_to_path: dict[str, str] = {}
        self._path_to_token: dict[str, str] = {}

        # Lazy-start: cleanup thread is created on first acquire to avoid
        # accumulating orphan threads during testing / repeated resets.
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._thread_mu = threading.Lock()  # guards _cleanup_thread init  # leaf lock: never held while acquiring a LockLevel lock

    def _ensure_cleanup_thread(self) -> None:
        """Start the background cleanup daemon on first use (double-checked locking)."""
        if self._cleanup_thread is not None:
            return
        with self._thread_mu:
            if self._cleanup_thread is not None:
                return
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="RepoLockCleanup",
                daemon=True,
            )
            self._cleanup_thread.start()

    # ------------------------------------------------------------------
    # Opaque token API (card-action security)
    # ------------------------------------------------------------------

    # Maximum number of cached token ↔ path mappings.  When exceeded,
    # orphan mappings (no active lock) are evicted before adding new ones.
    _TOKEN_MAP_CAPACITY = 1024

    def path_to_token(self, root_path: str) -> str:
        """Return a deterministic opaque token for *root_path*.

        The token is a SHA-256 hex prefix (32 chars) of the normalized path.
        Mappings are cached so ``token_to_path`` can reverse them.
        Thread-safe — protected by ``_mu``.

        When the mapping cache exceeds ``_TOKEN_MAP_CAPACITY``, orphan
        entries (paths without an active lock) are purged first.  If still
        over capacity the new token is returned without caching.
        """
        key = normalize_repo_path(root_path)
        if key is None:
            return ""
        with self._mu:
            existing = self._path_to_token.get(key)
            if existing:
                return existing
            token = hashlib.sha256(key.encode()).hexdigest()[:32]
            # Enforce capacity: purge orphan mappings if over limit.
            if len(self._token_to_path) >= self._TOKEN_MAP_CAPACITY:
                orphans = [
                    p for p in list(self._path_to_token)
                    if p not in self._locks
                ]
                for p in orphans:
                    t = self._path_to_token.pop(p, None)
                    if t is not None:
                        self._token_to_path.pop(t, None)
            # If still over capacity after purge, return without caching.
            if len(self._token_to_path) >= self._TOKEN_MAP_CAPACITY:
                return token
            self._path_to_token[key] = token
            self._token_to_path[token] = key
            return token

    def token_to_path(self, token: str) -> Optional[str]:
        """Reverse-lookup: return the normalized path for *token*, or ``None``."""
        if not token:
            return None
        with self._mu:
            return self._token_to_path.get(token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def heartbeat_interval(self) -> float:
        """Renew active leases at least three times before they expire."""

        lease_seconds = float(self._hard_timeout)
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            return _DEFAULT_HEARTBEAT_INTERVAL_S
        return min(
            _DEFAULT_HEARTBEAT_INTERVAL_S,
            lease_seconds / _HEARTBEAT_RENEWALS_PER_LEASE,
        )

    def acquire(
        self,
        root_path: str,
        chat_id: str,
        is_p2p: bool = False,
        *,
        sender_id: str = "",
        owner_id: str | None = None,
    ) -> AcquireResult:
        """Try to acquire a lock for one chat/request owner.

        本方法支持**可重入**（reentrant）语义：

        - ``is_p2p=True`` → 始终成功（私聊特权，不实际获取锁）。
        - 同一 ``(chat_id, owner_id)`` 重复获取 → refcount++，成功。调用方需对应
          调用 ``release()`` 来递减 refcount，降为 0 时释放。
        - 不同 owner 持有 → 拒绝，返回持有聊天信息。

        ``owner_id=None`` preserves the legacy chat-scoped API for callers
        outside the task scheduler. Production request paths pass the current
        scheduler run id explicitly.
        """
        key = normalize_repo_path(root_path)
        if key is None:
            return AcquireResult(success=True)

        if is_p2p:
            logger.debug("RepoLock: p2p bypass for %s (chat=%s)", key, chat_id[:12])
            return AcquireResult(success=True)

        self._ensure_cleanup_thread()

        now = time.monotonic()
        normalized_owner_id = str(owner_id or "")
        with self._mu:
            entry = self._locks.get(key)
            if entry is None:
                # No one holds this lock — create new entry.
                self._locks[key] = RepoLockEntry(
                    chat_id=chat_id,
                    owner_id=normalized_owner_id,
                    acquired_at=now,
                    last_active_time=now,
                    last_sender_id=sender_id,
                )
                logger.info("RepoLock: acquired %s by chat=%s", key, chat_id[:12])
                return AcquireResult(success=True)

            if (
                entry.chat_id == chat_id
                and entry.owner_id == normalized_owner_id
            ):
                # Reentrant — same request owner, bump refcount + refresh.
                entry.refcount += 1
                entry.last_active_time = now
                entry.last_sender_id = sender_id
                logger.debug("RepoLock: reentrant %s chat=%s refcount=%d", key, chat_id[:12], entry.refcount)
                return AcquireResult(success=True)

            # Conflict — another chat holds the lock.
            logger.warning(
                "RepoLock: conflict %s requested by chat=%s, held by chat=%s",
                key, chat_id[:12], entry.chat_id[:12],
            )
            # Track blocked chat for release notification (bounded to 10).
            blocked = self._blocked_chats.setdefault(key, set())
            if len(blocked) < 10:
                blocked.add(chat_id)
            return AcquireResult(
                success=False,
                holder_chat_id=entry.chat_id,
                locked_since=entry.acquired_at,
                last_active_time=entry.last_active_time,
            )

    def release(
        self,
        root_path: str,
        chat_id: str,
        *,
        owner_id: str | None = None,
    ) -> None:
        """Decrement refcount; remove entry when it reaches 0."""
        key = normalize_repo_path(root_path)
        if key is None:
            return

        notify_chats: set[str] = set()
        normalized_owner_id = str(owner_id or "")
        with self._mu:
            entry = self._locks.get(key)
            if (
                entry is None
                or entry.chat_id != chat_id
                or entry.owner_id != normalized_owner_id
            ):
                return
            entry.refcount -= 1
            if entry.refcount <= 0:
                del self._locks[key]
                # Clean up opaque token mappings (symmetric with _cleanup_idle).
                token = self._path_to_token.pop(key, None)
                if token is not None:
                    self._token_to_path.pop(token, None)
                notify_chats = self._blocked_chats.pop(key, set())
                logger.info("RepoLock: released %s by chat=%s", key, chat_id[:12])
            else:
                entry.last_active_time = time.monotonic()

        # Notify blocked chats outside the lock.
        if notify_chats:
            self.on_release.emit(key, notify_chats)

    def force_release(self, root_path: str) -> None:
        """Forcefully release the lock regardless of holder (admin use)."""
        key = normalize_repo_path(root_path)
        if key is None:
            return
        notify_chats: set[str] = set()
        with self._mu:
            entry = self._locks.pop(key, None)
            if entry:
                # Clean up opaque token mappings (symmetric with _cleanup_idle).
                token = self._path_to_token.pop(key, None)
                if token is not None:
                    self._token_to_path.pop(token, None)
                notify_chats = self._blocked_chats.pop(key, set())
                logger.warning("RepoLock: force-released %s (was held by chat=%s)", key, entry.chat_id[:12])

        # Notify blocked chats outside the lock.
        if notify_chats:
            self.on_release.emit(key, notify_chats)

    def touch(
        self,
        root_path: str,
        chat_id: str,
        *,
        owner_id: str | None = None,
    ) -> None:
        """Refresh ``last_active_time`` to prevent idle-timeout release."""
        key = normalize_repo_path(root_path)
        if key is None:
            return
        normalized_owner_id = str(owner_id or "")
        with self._mu:
            entry = self._locks.get(key)
            if (
                entry
                and entry.chat_id == chat_id
                and entry.owner_id == normalized_owner_id
            ):
                entry.last_active_time = time.monotonic()

    def list_locks(self) -> list[RepoLockInfo]:
        """Return a snapshot of all active locks (for diagnostics)."""
        now = time.monotonic()
        with self._mu:
            return [
                RepoLockInfo(
                    root_path=path,
                    chat_id=entry.chat_id,
                    refcount=entry.refcount,
                    acquired_at=entry.acquired_at,
                    last_active_time=entry.last_active_time,
                    idle_seconds=now - entry.last_active_time,
                    last_sender_id=entry.last_sender_id,
                )
                for path, entry in self._locks.items()
            ]

    def get_lock_info(self, root_path: str) -> Optional[RepoLockInfo]:
        """Return lock info for a specific repo path, or None if not locked.

        O(1) dict lookup after path normalization.
        """
        key = normalize_repo_path(root_path)
        if key is None:
            return None
        now = time.monotonic()
        with self._mu:
            entry = self._locks.get(key)
            if entry is None:
                return None
            return RepoLockInfo(
                root_path=key,
                chat_id=entry.chat_id,
                refcount=entry.refcount,
                acquired_at=entry.acquired_at,
                last_active_time=entry.last_active_time,
                idle_seconds=now - entry.last_active_time,
                last_sender_id=entry.last_sender_id,
            )

    def is_held_by(self, root_path: str, chat_id: str) -> bool:
        """Check whether the repo lock for *root_path* is currently held by *chat_id*.

        Thread-safe: reads ``_locks`` under ``_mu``.
        Returns ``False`` if the path is invalid or not locked.
        """
        key = normalize_repo_path(root_path)
        if key is None:
            return False
        with self._mu:
            entry = self._locks.get(key)
            if entry is None:
                return False
            return entry.chat_id == chat_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def hold(
        self,
        root_path: str,
        chat_id: str,
        is_p2p: bool = False,
        on_conflict: Optional[Callable[[AcquireResult], None]] = None,
        *,
        sender_id: str = "",
        owner_id: str | None = None,
    ) -> Generator[AcquireResult, None, None]:
        """Context manager that acquires and auto-releases a repo lock.

        Usage::

            with repo_lock_mgr.hold(path, cid) as result:
                do_work()  # result.success is always True here

        When the lock cannot be acquired:

        - If *on_conflict* is ``None`` (default), ``LockConflictError`` is
          raised immediately — the ``with`` block body is **never entered**.
        - If *on_conflict* is provided, it is called with the failed
          ``AcquireResult`` and the result is yielded with
          ``success=False``.  The caller must check ``result.success``.
        """
        result = self.acquire(
            root_path,
            chat_id,
            is_p2p=is_p2p,
            sender_id=sender_id,
            owner_id=owner_id,
        )
        if not result.success:
            if on_conflict is not None:
                on_conflict(result)
                yield result
                return
            # Default: raise immediately so callers don't need a success check
            raise LockConflictError(
                f"Repo lock conflict for {root_path!r} (held by {result.holder_chat_id})",
                holder_chat_id=result.holder_chat_id or "",
                locked_since=result.locked_since or 0.0,
                root_path=root_path,
                last_active_time=result.last_active_time or 0.0,
            )

        try:
            yield result
        finally:
            # Only release for non-p2p — p2p never actually acquired.
            if not is_p2p:
                self.release(
                    root_path,
                    chat_id,
                    owner_id=owner_id,
                )

    # ------------------------------------------------------------------
    # Background cleanup
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Daemon loop: periodically evict idle locks."""
        while not self._stop_event.wait(timeout=self._cleanup_interval):
            try:
                self._cleanup_idle()
            except Exception:
                logger.error("RepoLock: cleanup error", exc_info=True)

    def _cleanup_idle(self) -> None:
        now = time.monotonic()
        hard_reclaimed: list[tuple[str, str]] = []  # (path, chat_id) for callback
        released_blocked: list[tuple[str, set[str]]] = []  # (path, blocked_chat_ids)

        with self._mu:
            # Phase 1: idle_timeout — only evict entries with refcount <= 0
            idle_expired = [
                path for path, entry in self._locks.items()
                if entry.refcount <= 0 and (now - entry.last_active_time) > self._idle_timeout
            ]
            for path in idle_expired:
                entry = self._locks.pop(path)
                token = self._path_to_token.pop(path, None)
                if token is not None:
                    self._token_to_path.pop(token, None)
                blocked = self._blocked_chats.pop(path, set())
                if blocked:
                    released_blocked.append((path, blocked))
                logger.warning(
                    "RepoLock: idle-timeout released %s (chat=%s, idle=%.0fs)",
                    path, entry.chat_id[:12], now - entry.last_active_time,
                )

            # Phase 2: hard_timeout — reclaim only an active entry whose
            # renewable lease has stopped receiving touch() heartbeats.
            # Total runtime is deliberately not a cap: programming tasks may
            # validly run longer than this threshold, and reclaiming a recently
            # renewed lock would allow another request to mutate the same repo.
            hard_expired = [
                path for path, entry in self._locks.items()
                if (
                    entry.refcount > 0
                    and (now - entry.last_active_time) > self._hard_timeout
                )
            ]
            for path in hard_expired:
                entry = self._locks.pop(path)
                token = self._path_to_token.pop(path, None)
                if token is not None:
                    self._token_to_path.pop(token, None)
                blocked = self._blocked_chats.pop(path, set())
                if blocked:
                    released_blocked.append((path, blocked))
                logger.critical(
                    "RepoLock: hard-timeout force-reclaimed %s "
                    "(chat=%s, refcount=%d, lease_idle=%.0fs, held=%.0fs)",
                    path,
                    entry.chat_id[:12],
                    entry.refcount,
                    now - entry.last_active_time,
                    now - entry.acquired_at,
                )
                hard_reclaimed.append((path, entry.chat_id))

            # Phase 3: warn about refcount > 0 entries that are idle but not
            # yet hard-expired (potential missing touch() calls).
            for path, entry in self._locks.items():
                if entry.refcount > 0 and (now - entry.last_active_time) > self._idle_timeout:
                    logger.warning(
                        "RepoLock: active lock idle without touch %s (chat=%s, refcount=%d, idle=%.0fs) — not evicted",
                        path, entry.chat_id[:12], entry.refcount, now - entry.last_active_time,
                    )

            # Phase 4: clean orphan token mappings — tokens whose paths have
            # no corresponding entry in _locks (e.g. path_to_token() was
            # called for card-building but the lock was released separately).
            orphan_paths = [
                p for p in self._path_to_token
                if p not in self._locks
            ]
            for p in orphan_paths:
                token = self._path_to_token.pop(p, None)
                if token is not None:
                    self._token_to_path.pop(token, None)

        # Fire-and-forget: notify holders of hard-timeout reclaimed locks
        # (outside _mu to avoid blocking other threads during I/O).
        for path, chat_id in hard_reclaimed:
            self.on_reclaim.emit(path, chat_id)

        # Notify blocked chats about released locks.
        for path, blocked in released_blocked:
            self.on_release.emit(path, blocked)

    def register_hard_timeout_callback(
        self, callback: Callable[[str, str], None],
    ) -> None:
        """Register a hard-timeout reclaim callback (compat shim).

        Prefer ``on_reclaim.subscribe(callback)`` for new code.
        This method exists for backward compatibility — it subscribes the
        callback via :attr:`on_reclaim`.
        """
        self.on_reclaim.subscribe(callback)

    def shutdown(self) -> None:
        """Stop the cleanup thread (for testing / graceful shutdown)."""
        self._stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[RepoLockManager] = None
_instance_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def get_repo_lock_manager(
    *,
    on_hard_timeout_reclaim: Optional[Callable[[str, str], None]] = None,
) -> RepoLockManager:
    """Return the global RepoLockManager singleton (lazy-initialized).

    Parameters
    ----------
    on_hard_timeout_reclaim:
        Optional callback forwarded to the constructor on first creation.
        If the singleton already exists and a callback is provided, it is
        registered via :meth:`RepoLockManager.register_hard_timeout_callback`.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = RepoLockManager(
                    on_hard_timeout_reclaim=on_hard_timeout_reclaim,
                )
                return _instance
    if on_hard_timeout_reclaim is not None:
        _instance.register_hard_timeout_callback(on_hard_timeout_reclaim)
    return _instance


def shutdown_if_active() -> None:
    """Best-effort shutdown of the singleton if it was ever created.

    Safe to call multiple times (idempotent) and even before the singleton
    is initialized — in that case it simply does nothing.
    """
    if _instance is not None:
        try:
            _instance.shutdown()
        except Exception:
            logger.debug("RepoLockManager shutdown error", exc_info=True)


def _reset_repo_lock_manager_for_testing() -> None:
    """Reset the singleton. **Test-only.**"""
    global _instance
    with _instance_lock:
        if _instance is not None:
            _instance.shutdown()
        _instance = None
