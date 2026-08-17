"""Unified tool registry for Workflow Engine.

Discovers available tools from ACP providers at runtime,
replacing the hardcoded TOOL_DESCRIPTIONS constant.

Usage:
    from .tool_registry import get_available_tools

    tools = get_available_tools(require_available=True)
    # -> {"traex": "高并发推理·轻量任务", "claude": "Anthropic 深度推理", ...}
"""

from __future__ import annotations

import logging
import shutil
import threading
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_CACHE_TTL_S: float = 300.0  # 5-minute cache
_cache_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_cached_tools: dict[str, str] | None = None
_cached_at: float = 0.0
_cached_selectable_tools: dict[str, str] | None = None
_cached_selectable_at: float = 0.0

# ---------------------------------------------------------------------------
# Fallback descriptions (used when provider probing fails)
# ---------------------------------------------------------------------------

_FALLBACK_DESCRIPTIONS: dict[str, str] = {
    "coco": "全栈编程·支持 subagent",
    "aiden": "代码审查·架构设计",
    "codex": "OpenAI 自主编程",
    "claude": "Anthropic 深度推理",
    "traex": "高并发推理·轻量任务",
    "gemini": "Google 多模态推理",
    "grok": "xAI 原生 ACP 编程",
    "dsh": "DeepSeek Harness 原生 ACP 编程",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_available_tools(
    *,
    force_refresh: bool = False,
    require_available: bool = False,
) -> dict[str, str]:
    """Return a dict of tool_name -> description for all available tools.

    By default this returns known Workflow-capable ACP tools, including
    fallback names for compatibility. ``require_available=True`` is for
    user-facing selection cards: it probes provider availability and returns
    only tools that can actually run in the current environment.

    Thread-safe. On failure, returns the fallback dict.
    """
    global _cached_selectable_at, _cached_selectable_tools, _cached_tools, _cached_at

    if require_available:
        now = time.monotonic()
        if not force_refresh and _cached_selectable_tools is not None and (now - _cached_selectable_at) < _CACHE_TTL_S:
            return dict(_cached_selectable_tools)

        with _cache_lock:
            if (
                not force_refresh
                and _cached_selectable_tools is not None
                and (time.monotonic() - _cached_selectable_at) < _CACHE_TTL_S
            ):
                return dict(_cached_selectable_tools)

            tools = _discover_tools(require_available=True)
            _cached_selectable_tools = tools
            _cached_selectable_at = time.monotonic()
            return dict(tools)

    now = time.monotonic()
    if not force_refresh and _cached_tools is not None and (now - _cached_at) < _CACHE_TTL_S:
        return dict(_cached_tools)

    with _cache_lock:
        # Double-check after acquiring lock
        if not force_refresh and _cached_tools is not None and (time.monotonic() - _cached_at) < _CACHE_TTL_S:
            return dict(_cached_tools)

        tools = _discover_tools(require_available=False)
        _cached_tools = tools
        _cached_at = time.monotonic()
        return dict(tools)


def invalidate_cache() -> None:
    """Force the next get_available_tools() call to re-discover."""
    global _cached_selectable_at, _cached_selectable_tools, _cached_tools, _cached_at
    with _cache_lock:
        _cached_tools = None
        _cached_at = 0.0
        _cached_selectable_tools = None
        _cached_selectable_at = 0.0


# ---------------------------------------------------------------------------
# Discovery implementation
# ---------------------------------------------------------------------------


def _discover_tools(*, require_available: bool = False) -> dict[str, str]:
    """Discover Workflow tools.

    ``require_available`` is intentionally strict: no fallback-only names are
    added, so selection UIs cannot offer tools missing from the current host.
    """
    result: dict[str, str] = {}

    # ACP providers. In selectable mode this runs provider availability probes
    # such as ``which`` / ``--help`` checks through the shared registry cache.
    try:
        result.update(_discover_acp_tools(require_available=require_available))
    except Exception as e:
        logger.debug("ACP tool discovery failed, using fallback: %s", repr(e))
        if not require_available:
            for name in ("traex", "claude", "codex", "grok", "dsh", "aiden", "gemini", "coco"):
                if name not in result:
                    result[name] = _FALLBACK_DESCRIPTIONS.get(name, name)

    # Ensure at least the fallback set is covered
    if not require_available:
        for name, desc in _FALLBACK_DESCRIPTIONS.items():
            if name not in result:
                result[name] = desc

    return result


def _discover_acp_tools(*, require_available: bool = False) -> dict[str, str]:
    """Discover tools from ACP providers."""
    tools: dict[str, str] = {}

    try:
        from ..acp.helper import is_programming_tool_available
        from ..acp.providers import get_providers
    except ImportError:
        logger.debug("Cannot import ACP providers module")
        return tools

    try:
        providers = get_providers()
    except Exception as e:
        logger.debug("get_providers() failed: %s", e)
        return tools

    # Try to get localized descriptions from text module
    desc_map: dict[str, str] = {}
    try:
        from ..utils.text import get_acp_result_header_text
        headers = get_acp_result_header_text()
        for name in providers:
            key = f"tool_desc_{name}"
            if key in headers:
                desc_map[name] = headers[key]
    except Exception:
        pass

    for name in providers:
        if require_available:
            available = False

            # User-facing selection cards need a fast answer. A binary on PATH
            # is enough to offer the tool; the actual ACP startup path still
            # performs its normal capability checks and fallbacks later.
            if shutil.which(name) is not None:
                available = True
                try:
                    is_programming_tool_available(
                        name,
                        allow_sync_probe=False,
                        trigger_async_probe=True,
                    )
                except Exception:
                    logger.debug("Async ACP availability warmup failed for %s", name, exc_info=True)
            else:
                get_fallback = getattr(providers.get(name), "get_fallback_command", None)
                try:
                    available = bool(callable(get_fallback) and get_fallback())
                except Exception:
                    available = False
                if not available:
                    try:
                        available = is_programming_tool_available(
                            name,
                            allow_sync_probe=False,
                            trigger_async_probe=True,
                        )
                    except Exception:
                        available = False

            if not available:
                continue
        description = (
            desc_map.get(name)
            or _FALLBACK_DESCRIPTIONS.get(name)
            or name
        )
        tools[name] = description

    return tools
