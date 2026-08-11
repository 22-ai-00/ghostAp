"""ACP Provider 协议与工具注册中心。

该模块提供统一的 ACP Provider 抽象层（Provider/Registry 机制），
用于替代硬编码的条件分支，让“工具类型”与“启动/探测逻辑”解耦。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Iterable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ACPProvider(Protocol):
    """ACP Provider 协议接口。

    约定：
    - `name` 为工具的规范名（如 coco/claude/aiden/codex）。
    - `check_availability()` 仅做轻量探测：判断二进制是否存在、是否支持 `acp serve`。
    - `get_serve_command()` 负责返回启动 ACP Server 的 (cmd, args)。
    - `get_fallback_command()` 可选：在不可用时提供可执行的兜底命令。
    - `normalize_model_name()` 可选：把 UI/配置层模型值转换为后端可执行模型值。
    """

    @property
    def name(self) -> str: ...

    @property
    def skip_model_selection(self) -> bool: ...

    def get_serve_command(self, model_name: Optional[str] = None) -> tuple[str, list[str]]: ...

    def check_availability(self) -> bool: ...

    def get_fallback_command(self, model_name: Optional[str] = None) -> Optional[tuple[str, list[str]]]: ...

    def normalize_model_name(self, model_name: Optional[str] = None) -> Optional[str]: ...


class ToolRegistry:
    """ACP Provider 注册中心。

    目标：
    - 统一注册/提取 Provider
    - 通过 Provider 生成启动命令，避免上层硬编码分支
    - 提供最佳努力的可用性缓存与后台预热（更高级的 LRU/异步策略在后续任务完善）
    """

    def __init__(self):
        self._providers: dict[str, ACPProvider] = {}
        self._default_provider: Optional[str] = None
        # availability LRU cache: name -> (available, ts)
        self._availability_cache: "OrderedDict[str, tuple[bool, float]]" = OrderedDict()
        self._availability_cache_maxsize: int = 64
        self._availability_cache_ttl_s: float = 60.0
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._preheated: bool = False
        self._probe_inflight: set[str] = set()

    def register(self, provider: ACPProvider, is_default: bool = False) -> None:
        """Register a new ACP provider."""
        name = str(provider.name or "").lower()
        if not name:
            raise ValueError("provider.name 不能为空")
        with self._lock:
            self._providers[name] = provider
            if is_default or not self._default_provider:
                self._default_provider = name
        logger.debug("[ACP] Registered provider: %s", name)

    def get_provider(self, name: str) -> Optional[ACPProvider]:
        """Retrieve a registered provider by name."""
        key = str(name or "").lower()
        if not key:
            return None
        with self._lock:
            return self._providers.get(key)

    def get_cached_availability(self, name: str) -> Optional[bool]:
        """Return fresh cached availability, or None when cache is missing/stale."""
        key = str(name or "").lower()
        if not key:
            return None
        now = time.time()
        with self._lock:
            v = self._availability_cache.get(key)
            if v is None:
                return None
            ok, ts = v
            if (now - float(ts or 0.0)) > float(self._availability_cache_ttl_s or 0.0):
                try:
                    self._availability_cache.pop(key, None)
                except Exception:
                    logger.debug("failed to pop expired cache entry", exc_info=True)
                return None
            try:
                self._availability_cache.move_to_end(key)
            except Exception:
                logger.debug("LRU move_to_end failed", exc_info=True)
            return bool(ok)

    def get_availability(
        self,
        name: str,
        *,
        allow_sync_probe: bool = False,
        trigger_async_probe: bool = True,
    ) -> bool:
        """Get tool availability with explicit latency control.

        Defaults are optimized for user-facing status views:
        - prefer fresh cache
        - never block on subprocess probes on cache miss
        - trigger async probe to refresh cache in background
        """
        key = str(name or "").lower()
        if not key:
            return False
        provider = self.get_provider(key)
        if not provider:
            return False

        cached = self.get_cached_availability(key)
        if cached is not None:
            return bool(cached)

        if trigger_async_probe:
            self._probe_availability_async(key)

        if not allow_sync_probe:
            return False

        try:
            ok = bool(provider.check_availability())
        except Exception:
            ok = False
        self._set_availability_cache(key, ok)
        return ok

    def _check_availability_cached(self, provider: ACPProvider) -> bool:
        name = str(provider.name or "").lower()
        if not name:
            return False

        now = time.time()
        with self._lock:
            v = self._availability_cache.get(name)
            if v is not None:
                ok, ts = v
                if (now - float(ts or 0.0)) <= float(self._availability_cache_ttl_s or 0.0):
                    # LRU touch
                    try:
                        self._availability_cache.move_to_end(name)
                    except Exception:
                        logger.debug("LRU move_to_end failed", exc_info=True)
                    return bool(ok)
                # expired
                try:
                    self._availability_cache.pop(name, None)
                except Exception:
                    logger.debug("failed to pop expired cache entry", exc_info=True)

        # cache miss/expired: do a real check (may be expensive; avoid for hot tools in get_serve_command)
        available = bool(provider.check_availability())
        self._set_availability_cache(name, available)
        return available

    def _set_availability_cache(self, name: str, available: bool) -> None:
        key = str(name or "").lower()
        if not key:
            return
        now = time.time()
        with self._lock:
            self._availability_cache[key] = (bool(available), now)
            try:
                self._availability_cache.move_to_end(key)
            except Exception:
                logger.debug("LRU move_to_end failed", exc_info=True)
            # evict
            while len(self._availability_cache) > int(self._availability_cache_maxsize or 64):
                try:
                    self._availability_cache.popitem(last=False)
                except Exception:
                    break

    def _probe_availability_async(self, name: str) -> None:
        """后台探活：不阻塞主流程。"""
        key = str(name or "").lower()
        if not key:
            return
        with self._lock:
            if key in self._probe_inflight:
                return
            self._probe_inflight.add(key)

        def _run():
            try:
                p = self.get_provider(key)
                if not p:
                    return
                ok = bool(p.check_availability())
                self._set_availability_cache(key, ok)
            except Exception:
                # best-effort
                pass
            finally:
                with self._lock:
                    self._probe_inflight.discard(key)

        t = threading.Thread(target=_run, name=f"acp-probe-{key}", daemon=True)
        t.start()

    def get_serve_command(self, name: str, model_name: Optional[str] = None) -> tuple[str, list[str]]:
        """Get the serve command for the specified tool.

        If the tool is registered, delegates to its provider.
        If not registered, falls back to a generic (name, ["acp", "serve"]) implementation.
        """
        provider = self.get_provider(name)
        if provider:
            tool_name = str(provider.name or "").lower()

            # Performance: 对 coco/aiden 这类启动链路敏感的工具，避免在解析阶段做阻塞探活。
            # - 若缓存命中则按缓存裁决
            # - 若缓存未命中：先返回启动命令（乐观），并在后台异步探活填充缓存
            hot_tools = {"coco", "aiden", "codex", "traex"}
            if tool_name in hot_tools:
                # cached decision (best-effort)
                now = time.time()
                cached_negative = False
                with self._lock:
                    v = self._availability_cache.get(tool_name)
                    if v is not None:
                        ok, ts = v
                        if (now - float(ts or 0.0)) <= float(self._availability_cache_ttl_s or 0.0):
                            try:
                                self._availability_cache.move_to_end(tool_name)
                            except Exception:
                                logger.debug("LRU move_to_end failed", exc_info=True)
                            if bool(ok):
                                return provider.get_serve_command(model_name)
                            cached_negative = True

                if cached_negative:
                    # Negative cache can be stale (e.g. transient probe failure,
                    # tool upgraded after startup, or prior background probe ran in a
                    # degraded environment). Re-check once synchronously before failing.
                    fresh_ok = bool(provider.check_availability())
                    self._set_availability_cache(tool_name, fresh_ok)
                    if fresh_ok:
                        return provider.get_serve_command(model_name)

                    fb = provider.get_fallback_command(model_name)
                    if fb:
                        return fb
                    raise RuntimeError(f"Tool '{name}' is registered but not available for ACP mode.")

                # cache miss: schedule probe and return optimistic serve command
                self._probe_availability_async(tool_name)
                return provider.get_serve_command(model_name)

            # Non-hot tools: keep deterministic behavior
            if self._check_availability_cached(provider):
                return provider.get_serve_command(model_name)
            fallback = provider.get_fallback_command(model_name)
            if fallback:
                return fallback
            raise RuntimeError(f"Tool '{name}' is registered but not available for ACP mode.")

        # Generic fallback for unregistered but potentially valid commands
        logger.debug("[ACP] Tool '%s' not registered in ToolRegistry, using generic fallback.", name)
        return name, ["acp", "serve"]

    # -------------------------
    # Performance: async preheat
    # -------------------------
    def preheat_async(self, names: Optional[list[str]] = None) -> None:
        """Best-effort async preheat of availability cache.

        - Spawns a daemon thread that probes provider.check_availability()
          for the given names (or common tools if None).
        - Never raises; safe to call multiple times (deduped by flag).
        """
        try:
            with self._lock:
                if self._preheated:
                    return
                self._preheated = True

            targets = [n.lower() for n in (names or ["coco", "aiden", "codex", "traex", "grok"])]
            # fan-out probes
            for n in targets:
                try:
                    self._probe_availability_async(n)
                except Exception:
                    continue
        except Exception:
            # best-effort: completely ignore preheat failures
            pass

    # ----------------------------------
    # Test-only: guarded state reset
    # ----------------------------------
    def _reset_for_testing(self, names: Iterable[str]) -> None:
        """Remove specified providers under *self._lock*.  **Test-only.**

        After removal, if ``_default_provider`` no longer exists in the
        remaining providers it is set to ``None``.
        """
        with self._lock:
            for n in names:
                self._providers.pop(n, None)
            if self._default_provider and self._default_provider not in self._providers:
                self._default_provider = None


# 兼容别名：旧测试/调用点可能仍引用 `_ToolRegistry`
_ToolRegistry = ToolRegistry


# 全局注册中心单例
tool_registry = ToolRegistry()
