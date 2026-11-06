"""Process-level LRU cache for Feishu user display names.

Resolves ``user_id`` → ``display_name`` by calling the Feishu contact API
on cache miss, with a configurable capacity and TTL.  Thread-safe.

Degradation strategy: stable permission rejections (for example missing scopes)
are negatively cached for one hour, while transient/local failures use 60 s.
The caller always receives ``user_id[:8] + "(ID)"`` as a fallback.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.utils.thread_pools import submit_io

logger = logging.getLogger(__name__)

_CACHE_CAPACITY = 500
_TTL_SECONDS = 3600  # 1 hour
_NEGATIVE_TTL_SECONDS = 60  # cache failures for 60 s
_API_FAILURE_TTL_SECONDS = 3600  # stable server rejection (for example missing scope)
_MAX_PENDING_REFRESHES = 32
_PERMISSION_ERROR_CODE_PREFIX = "999916"


class _CacheEntry:
    __slots__ = ("display_name", "expires_at")

    def __init__(self, display_name: str, ttl: float):
        self.display_name = display_name
        self.expires_at = time.monotonic() + ttl


@dataclass(frozen=True)
class _FetchResult:
    display_name: str
    ttl_seconds: float


_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
_cache_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_refreshing: set[str] = set()


def _fallback_name(user_id: str) -> str:
    return (user_id[:8] + "(ID)") if user_id else ""


def _negative_ttl_for_response(response: Any) -> float:
    """Use a long TTL only for Feishu's stable permission-error family."""
    code = getattr(response, "code", None)
    if str(code).startswith(_PERMISSION_ERROR_CODE_PREFIX):
        return _API_FAILURE_TTL_SECONDS
    return _NEGATIVE_TTL_SECONDS


def _get_cached(user_id: str, now: float) -> str | None:
    with _cache_lock:
        entry = _cache.get(user_id)
        if entry is None or entry.expires_at <= now:
            return None
        _cache.move_to_end(user_id)
        return entry.display_name


def _store_cached(
    user_id: str,
    display_name: str,
    *,
    ttl_seconds: float | None = None,
) -> None:
    ttl = ttl_seconds
    if ttl is None:
        ttl = (
            _TTL_SECONDS
            if display_name and not display_name.endswith("(ID)")
            else _NEGATIVE_TTL_SECONDS
        )
    with _cache_lock:
        _cache[user_id] = _CacheEntry(display_name, ttl)
        _cache.move_to_end(user_id)
        while len(_cache) > _CACHE_CAPACITY:
            _cache.popitem(last=False)


def resolve_display_name(
    user_id: str,
    api_client_factory: Optional[Callable[[], Any]] = None,
) -> str:
    """Resolve a Feishu user_id to a human-readable display name.

    Returns the cached name if available and not expired.
    On cache miss, calls the Feishu contact API via *api_client_factory*.
    Falls back to ``user_id[:8] + "(ID)"`` on any failure.
    """
    if not user_id:
        return ""

    cached = _get_cached(user_id, time.monotonic())
    if cached is not None:
        return cached

    # 2. Cache miss or expired — try API
    result = _fetch_display_name(user_id, api_client_factory)

    _store_cached(
        user_id,
        result.display_name,
        ttl_seconds=result.ttl_seconds,
    )

    return result.display_name


def resolve_display_name_nonblocking(
    user_id: str,
    api_client_factory: Optional[Callable[[], Any]] = None,
) -> str:
    """Return immediately and refresh a missing display name in the I/O pool.

    Display names are presentation-only metadata and must never delay message
    admission or routing.  A process-wide single-flight guard prevents repeated
    Contact API calls while one refresh is already running.
    """
    if not user_id:
        return ""
    cached = _get_cached(user_id, time.monotonic())
    if cached is not None:
        return cached
    fallback = _fallback_name(user_id)
    if api_client_factory is None:
        _store_cached(user_id, fallback)
        return fallback
    with _cache_lock:
        now = time.monotonic()
        entry = _cache.get(user_id)
        if entry is not None and entry.expires_at > now:
            _cache.move_to_end(user_id)
            return entry.display_name
        if user_id in _refreshing:
            return fallback
        if len(_refreshing) >= _MAX_PENDING_REFRESHES:
            _cache[user_id] = _CacheEntry(fallback, _NEGATIVE_TTL_SECONDS)
            _cache.move_to_end(user_id)
            while len(_cache) > _CACHE_CAPACITY:
                _cache.popitem(last=False)
            return fallback
        _refreshing.add(user_id)
    try:
        submit_io(_refresh_display_name, user_id, api_client_factory)
    except Exception:
        with _cache_lock:
            _refreshing.discard(user_id)
        logger.debug(
            "Failed to schedule display name refresh for user_id=%s",
            user_id[:12],
            exc_info=True,
        )
    return fallback


def _refresh_display_name(
    user_id: str,
    api_client_factory: Callable[[], Any],
) -> None:
    try:
        result = _fetch_display_name(user_id, api_client_factory)
        _store_cached(
            user_id,
            result.display_name,
            ttl_seconds=result.ttl_seconds,
        )
    finally:
        with _cache_lock:
            _refreshing.discard(user_id)


def _fetch_display_name(
    user_id: str,
    api_client_factory: Optional[Callable[[], Any]] = None,
) -> _FetchResult:
    """Call Feishu contact API to get display name.  Best-effort."""
    fallback = _fallback_name(user_id)

    if api_client_factory is None:
        return _FetchResult(fallback, _NEGATIVE_TTL_SECONDS)

    try:
        from lark_oapi.api.contact.v3 import GetUserRequest

        client = api_client_factory()
        req = GetUserRequest.builder().user_id(user_id).user_id_type("open_id").build()
        resp = client.contact.v3.user.get(req)
        if resp and resp.success() and resp.data and resp.data.user:
            name = resp.data.user.name or ""
            if name:
                return _FetchResult(name, _TTL_SECONDS)
        logger.debug(
            "Contact API returned an unsuccessful response for user_id=%s, code=%s",
            user_id[:12],
            getattr(resp, "code", None),
        )
        return _FetchResult(fallback, _negative_ttl_for_response(resp))
    except ImportError:
        logger.debug("lark_oapi not available for contact API; using fallback name")
        return _FetchResult(fallback, _NEGATIVE_TTL_SECONDS)
    except Exception:
        logger.debug("Failed to fetch display name for user_id=%s", user_id[:12], exc_info=True)
        return _FetchResult(fallback, _NEGATIVE_TTL_SECONDS)
