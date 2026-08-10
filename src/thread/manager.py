from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import Callable, Optional

from .models import ThreadContext

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 86400 * 7
_CLEANUP_INTERVAL = 3600


class ThreadContextManager:

    def __init__(self, ttl: float = _DEFAULT_TTL, cleanup_interval: float = _CLEANUP_INTERVAL, on_evict: Optional[Callable[[ThreadContext], None]] = None):
        self._contexts: dict[str, ThreadContext] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._ttl = ttl
        self._on_evict = on_evict
        self._cleanup_stop = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(cleanup_interval,),
            daemon=True,
            name="thread-ctx-cleanup",
        )
        self._cleanup_thread.start()

    def register(
        self,
        thread_root_id: str,
        chat_id: str,
        project_id: str,
        mode: str = "smart",
        tool_name: Optional[str] = None,
        model_name: Optional[str] = None,
        alias_keys: Optional[list[str]] = None,
    ) -> ThreadContext:
        ctx = ThreadContext(
            thread_root_id=thread_root_id,
            chat_id=chat_id,
            project_id=project_id,
            mode=mode,
            tool_name=tool_name,
            model_name=model_name,
        )
        aliases = tuple(
            dict.fromkeys(
                alias for alias in (alias_keys or []) if alias and alias != thread_root_id
            )
        )
        with self._lock:
            self._contexts[thread_root_id] = ctx
            self._aliases = {
                alias: root
                for alias, root in self._aliases.items()
                if root != thread_root_id
            }
            self._aliases.update({alias: thread_root_id for alias in aliases})
        alias_info = ",".join(aliases[:3]) if aliases else "none"
        logger.info(
            "[Thread] Registered: root=%s aliases=%s chat=%s project=%s mode=%s tool=%s model=%s",
            thread_root_id[:12],
            alias_info[:36],
            chat_id[:12],
            project_id,
            mode,
            tool_name,
            model_name,
        )
        return ctx

    def get(self, thread_root_id: str) -> Optional[ThreadContext]:
        with self._lock:
            ctx = self._contexts.get(self._aliases.get(thread_root_id, thread_root_id))
            if ctx:
                ctx.touch()
        return ctx

    def bind_engine(
        self,
        *,
        thread_root_id: str,
        chat_id: str,
        project_id: str,
        mode: str,
        tool_name: Optional[str] = None,
        model_name: Optional[str] = None,
        alias_keys: Optional[list[str]] = None,
    ) -> ThreadContext:
        return self.register(
            thread_root_id,
            chat_id,
            project_id,
            mode=mode,
            tool_name=tool_name,
            model_name=model_name,
            alias_keys=alias_keys,
        )

    def get_by_chat(self, chat_id: str) -> list[ThreadContext]:
        with self._lock:
            return [ctx for ctx in self._contexts.values() if ctx.chat_id == chat_id]

    def remove(self, thread_root_id: str) -> Optional[ThreadContext]:
        with self._lock:
            canonical = self._aliases.get(thread_root_id, thread_root_id)
            ctx = self._contexts.pop(canonical, None)
            self._aliases = {
                alias: root
                for alias, root in self._aliases.items()
                if root != canonical
            }
        if ctx and self._on_evict:
            try:
                self._on_evict(ctx)
            except Exception:
                logger.debug("[Thread] on_evict callback error", exc_info=True)
        return ctx

    def close(self) -> None:
        self._cleanup_stop.set()
        with self._lock:
            remaining = list(self._contexts.values())
            self._contexts.clear()
            self._aliases.clear()
        if self._on_evict:
            for ctx in remaining:
                try:
                    self._on_evict(ctx)
                except Exception:
                    logger.debug("[Thread] on_evict callback error", exc_info=True)

    def _cleanup_loop(self, interval: float) -> None:
        while not self._cleanup_stop.wait(timeout=interval):
            try:
                self._evict_expired()
            except Exception:
                logger.debug("[Thread] Cleanup error", exc_info=True)

    def _evict_expired(self) -> None:
        now = time.time()
        evicted: list[ThreadContext] = []
        with self._lock:
            expired = {
                root
                for root, ctx in self._contexts.items()
                if (now - ctx.last_active) > self._ttl
            }
            evicted = [self._contexts.pop(root) for root in expired]
            self._aliases = {
                alias: root
                for alias, root in self._aliases.items()
                if root not in expired
            }
        if evicted:
            logger.info("[Thread] Evicted %d expired thread contexts", len(evicted))
        if self._on_evict:
            for ctx in evicted:
                try:
                    self._on_evict(ctx)
                except Exception:
                    logger.debug("[Thread] on_evict callback error", exc_info=True)


_manager: Optional[ThreadContextManager] = None
_manager_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

_current_thread_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_thread_id", default=None)
_current_sender_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_sender_id", default=None)
_current_sender_union_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_sender_union_id", default=None
)
_current_sender_name: contextvars.ContextVar[str] = contextvars.ContextVar("current_sender_name", default="")
_current_is_p2p: contextvars.ContextVar[bool] = contextvars.ContextVar("current_is_p2p", default=False)
_current_tenant_key: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_tenant_key", default=None
)
_current_mentioned_names: contextvars.ContextVar[tuple[str, ...]] = (
    contextvars.ContextVar("current_mentioned_names", default=())
)


def get_thread_manager() -> ThreadContextManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ThreadContextManager()
    return _manager


def set_current_thread_id(thread_id: Optional[str]) -> None:
    _current_thread_id.set(thread_id)


def get_current_thread_id() -> Optional[str]:
    return _current_thread_id.get()


def set_current_sender_id(sender_id: Optional[str]) -> None:
    _current_sender_id.set(sender_id)


def get_current_sender_id() -> Optional[str]:
    return _current_sender_id.get()


def set_current_sender_union_id(sender_union_id: Optional[str]) -> None:
    _current_sender_union_id.set(sender_union_id)


def get_current_sender_union_id() -> Optional[str]:
    return _current_sender_union_id.get()


def set_current_is_p2p(is_p2p: bool) -> None:
    _current_is_p2p.set(is_p2p)


def get_current_is_p2p() -> bool:
    return _current_is_p2p.get()


def set_current_tenant_key(tenant_key: Optional[str]) -> None:
    _current_tenant_key.set(tenant_key)


def get_current_tenant_key() -> Optional[str]:
    return _current_tenant_key.get()


def set_current_mentioned_names(names: tuple[str, ...]) -> None:
    if not isinstance(names, tuple) or any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in names
    ):
        raise ValueError("mentioned names must be a normalized tuple")
    _current_mentioned_names.set(tuple(dict.fromkeys(names)))


def get_current_mentioned_names() -> tuple[str, ...]:
    return _current_mentioned_names.get()


def set_current_sender_name(name: str) -> None:
    _current_sender_name.set(name)


def get_current_sender_name() -> str:
    return _current_sender_name.get()
