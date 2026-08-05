
import dataclasses
import logging
import threading
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from acp.stdio import spawn_agent_process

from ..config import get_settings
from ..ttadk.models import ACPModelOption, ACPModelVariantOption, ACPToolOption
from ..utils.async_helpers import safe_wait_for
from ..utils.text import get_acp_result_header_text
from .client import GhostAPClient
from .providers import get_providers, tool_registry

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic ACP model probe cache for non-coco tools (Aiden, Codex, Gemini…).
#
# Three coordinated layers, all keyed by (tool_name, cwd) and guarded by a
# single leaf lock:
#   * positive cache  — a successful probe result, reused for its tool-specific TTL.
#   * negative cache  — remembers a recent empty/timed-out probe so a stuck tool
#                       (e.g. claude, which lacks ACP serve support and burns the
#                       full probe timeout every call) is not re-probed on every
#                       card click for _ACP_NEG_CACHE_TTL seconds.
#   * single-flight   — coalesces concurrent probes for the same key: the first
#                       thread runs the real probe, the rest wait on an Event and
#                       reuse the leader's result instead of each spawning their
#                       own `<tool> acp serve` subprocess (the "thundering herd"
#                       behind duplicate model_lookup log lines).
#
# Callers may mark a per-request default via `current_model`, so every value
# handed out is a deep-ish copy (fresh ACPModelOption instances); the shared
# cached objects are never mutated across callers.
# ---------------------------------------------------------------------------
_acp_probe_cache: dict[tuple[str, str], tuple[float, list[ACPModelOption]]] = {}
_acp_probe_cache_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_ACP_PROBE_CACHE_TTL = 300  # 5 minutes
_CODEX_PROBE_CACHE_TTL = 1800  # 30 minutes
_ACP_NEG_CACHE_TTL = 45  # remember "no models / timed out" briefly to avoid re-probing

# Negative cache: key -> timestamp of the failed probe.
_acp_neg_cache: dict[tuple[str, str], float] = {}
# Single-flight registry: key -> Event signalling the in-flight probe finished.
_acp_probe_inflight: dict[tuple[str, str], threading.Event] = {}
_acp_probe_generation: dict[tuple[str, str], int] = {}


def _positive_probe_cache_ttl(tool_name: str) -> float:
    return float(
        _CODEX_PROBE_CACHE_TTL
        if str(tool_name or "").strip().lower() == "codex"
        else _ACP_PROBE_CACHE_TTL
    )


def _probe_key(tool_name: str, cwd: Optional[str]) -> tuple[str, str]:
    """Normalized cache/single-flight key shared by all three layers."""
    return (str(tool_name or ""), str(cwd or ""))


def _copy_models(models: list[ACPModelOption]) -> list[ACPModelOption]:
    """Return fresh ACPModelOption instances so callers never mutate shared cache."""
    return [dataclasses.replace(m) for m in models]


def get_acp_model_cache_generation(
    tool_name: str,
    cwd: Optional[str] = None,
) -> int:
    key = _probe_key(tool_name, cwd)
    with _acp_probe_cache_lock:
        return _acp_probe_generation.get(key, 0)


def invalidate_acp_model_cache(
    tool_name: str,
    cwd: Optional[str] = None,
) -> None:
    key = _probe_key(tool_name, cwd)
    with _acp_probe_cache_lock:
        _acp_probe_cache.pop(key, None)
        _acp_neg_cache.pop(key, None)
        _acp_probe_generation[key] = _acp_probe_generation.get(key, 0) + 1


def _mark_default(models: list[ACPModelOption], current_model: Optional[str]) -> list[ACPModelOption]:
    """Re-mark is_default on a per-caller copy according to current_model."""
    if current_model:
        selected = str(current_model or "").strip()
        matched_model = next((m for m in models if m.name == selected), None)
        if matched_model is None:
            matched_model = next(
                (
                    m
                    for m in models
                    if any(
                        variant.name == selected
                        for variant in (
                            getattr(m, "selection_variants", ()) or ()
                        )
                    )
                ),
                None,
            )
        if matched_model is None:
            matched_model = next(
                (
                    m
                    for m in models
                    if any(
                        selected == f"{m.name}/{effort}"
                        for effort in (
                            getattr(m, "reasoning_efforts", ()) or ()
                        )
                    )
                ),
                None,
            )
        if matched_model is None:
            from .model_selection import split_codex_model_selection

            base_model, _effort = split_codex_model_selection(selected)
            matched_model = next(
                (
                    m
                    for m in models
                    if m.name == base_model
                    and bool(getattr(m, "reasoning_efforts", ()) or ())
                ),
                None,
            )
        if matched_model is None:
            return models
        for m in models:
            m.is_default = m is matched_model
    return models


def _get_cached_probe(tool_name: str, cwd: Optional[str] = None) -> list[ACPModelOption]:
    """Return a copy of the cached probe result if within TTL, else empty list."""
    key = _probe_key(tool_name, cwd)
    with _acp_probe_cache_lock:
        entry = _acp_probe_cache.get(key)
    if not entry:
        return []
    ts, models = entry
    if (_time.time() - ts) > _positive_probe_cache_ttl(tool_name):
        return []
    return _copy_models(models)


def _is_negatively_cached(tool_name: str, cwd: Optional[str] = None) -> bool:
    """True when a recent probe returned nothing and the neg-cache is still fresh."""
    key = _probe_key(tool_name, cwd)
    with _acp_probe_cache_lock:
        ts = _acp_neg_cache.get(key)
    if ts is None:
        return False
    if (_time.time() - ts) > _ACP_NEG_CACHE_TTL:
        with _acp_probe_cache_lock:
            # Drop only if unchanged, so a concurrent refresh isn't clobbered.
            if _acp_neg_cache.get(key) == ts:
                _acp_neg_cache.pop(key, None)
        return False
    return True


def _set_cached_probe(tool_name: str, models: list[ACPModelOption], cwd: Optional[str] = None) -> None:
    """Store a successful probe result and clear any stale negative marker."""
    key = _probe_key(tool_name, cwd)
    if not models:
        return
    with _acp_probe_cache_lock:
        _acp_probe_cache[key] = (_time.time(), _copy_models(models))
        _acp_neg_cache.pop(key, None)


def _set_negative_probe(tool_name: str, cwd: Optional[str] = None) -> None:
    """Remember that a probe just returned nothing (empty/timeout)."""
    key = _probe_key(tool_name, cwd)
    with _acp_probe_cache_lock:
        _acp_neg_cache[key] = _time.time()


def _local_fallback_models(
    tool_name: str, current_model: Optional[str] = None
) -> list[ACPModelOption]:
    """Return local model options for providers without reliable live model lists."""
    if current_model:
        return [
            ACPModelOption(
                name=str(current_model), description=str(current_model), is_default=True
            )
        ]
    return []


def list_acp_tools() -> list[ACPToolOption]:
    """List available ACP tools.

    使用共享文案层提供的工具描述文案，避免直接依赖旧 styles 聚合入口。
    """

    get_providers()
    names = ["coco", "claude", "aiden", "codex", "gemini", "traex"]
    out: list[ACPToolOption] = []
    headers = get_acp_result_header_text()

    for name in names:
        provider = tool_registry.get_provider(name)
        if not provider:
            continue
        try:
            available = bool(provider.check_availability())
        except Exception:
            logger.debug("[ACP] availability check failed for %s", name, exc_info=True)
            available = False
        if available:
            desc = headers.get(f"tool_desc_{name}") or name
            out.append(
                ACPToolOption(name=name, description=desc, is_default=(name == "coco"))
            )
    return out


def _probe_models_blocking(
    tool_name: str,
    cwd: Optional[str],
    current_model: Optional[str],
    probe_timeout: Optional[float],
) -> list[ACPModelOption]:
    """Run the real ACP model probe synchronously; return [] on timeout/failure."""
    try:
        timeout_s = _resolve_acp_model_probe_timeout(probe_timeout)
        from src.utils.async_helpers import run_async
        return run_async(
            safe_wait_for(
                probe_acp_models(tool_name, cwd, current_model),
                timeout=timeout_s,
                action=f"ACP {tool_name} 模型探测",
            )
        ) or []
    except Exception:
        logger.warning("[ACP] probe models failed for %s, will fallback", tool_name, exc_info=True)
        return []


def _fetch_non_coco_models_singleflight(
    tool_name: str,
    cwd: Optional[str],
    probe_timeout: Optional[float],
) -> list[ACPModelOption]:
    """Coalesce concurrent probes for one (tool, cwd): leader probes, waiters reuse.

    Returns fresh ACPModelOption copies (never shared cache objects). On any
    failure/timeout an empty list is returned and a negative-cache marker is set
    so the caller can degrade to a local fallback without re-probing.
    """
    key = _probe_key(tool_name, cwd)
    is_leader = False
    with _acp_probe_cache_lock:
        event = _acp_probe_inflight.get(key)
        if event is None:
            event = threading.Event()
            _acp_probe_inflight[key] = event
            is_leader = True

    if not is_leader:
        # Waiter: block on the leader instead of spawning a duplicate subprocess.
        wait_budget = _resolve_acp_model_probe_timeout(probe_timeout) + 5.0
        if event.wait(timeout=wait_budget):
            cached = _get_cached_probe(tool_name, cwd)  # already a copy
            if cached:
                logger.debug(
                    "[ACP] single-flight reuse for %s (%d models)", tool_name, len(cached)
                )
                return cached
        # Leader produced nothing (or is stuck): degrade via caller's fallback.
        return []

    # Leader: run the probe once and publish the result to waiters.
    try:
        models = _probe_models_blocking(tool_name, cwd, None, probe_timeout)
        if models:
            _set_cached_probe(tool_name, models, cwd)
            return _copy_models(models)
        _set_negative_probe(tool_name, cwd)
        return []
    finally:
        with _acp_probe_cache_lock:
            _acp_probe_inflight.pop(key, None)
        event.set()


def fetch_acp_models(
    tool_name: str,
    cwd: Optional[str],
    current_model: Optional[str] = None,
    probe_timeout: Optional[float] = None,
) -> list[ACPModelOption]:
    """Synchronous wrapper to probe available models from an ACP provider.

    For Coco, prefer the cached list maintained by ``CocoModelManager`` (the
    same source ``show_coco_status`` and the agent session bootstrap rely on).
    A successful probe there is cached for 5 minutes, so subsequent model
    selectors reuse the real ACP model list instead of degrading to the
    6-entry static ``DEFAULT_MODELS`` fallback.

    For non-Coco tools the live probe is guarded by three layers (positive
    cache, negative cache, single-flight) so concurrent card clicks and the
    background pre-heat thread never spawn duplicate ``<tool> acp serve``
    subprocesses, and a tool that cannot serve ACP (e.g. claude) is not
    re-probed on every interaction.
    """
    if tool_name == "coco":
        cached = _coco_models_from_manager(current_model)
        if cached:
            return cached
    else:
        cached = _get_cached_probe(tool_name, cwd)
        if cached:
            logger.debug("[ACP] using cached probe for %s (%d models)", tool_name, len(cached))
            return _mark_default(cached, current_model)
        if _is_negatively_cached(tool_name, cwd):
            logger.debug(
                "[ACP] negative-cache hit for %s, skipping live probe", tool_name
            )
            if tool_name == "codex":
                return []
            fallback = _local_fallback_models(tool_name, current_model)
            return fallback if fallback else []

        models = _fetch_non_coco_models_singleflight(
            tool_name, cwd, probe_timeout
        )
        if models:
            return _mark_default(models, current_model)
        if tool_name == "codex":
            return []
        fallback = _local_fallback_models(tool_name, current_model)
        return fallback if fallback else []

    # --- Coco-only path (unchanged): probe directly, then degrade via manager. ---
    models = _probe_models_blocking(tool_name, cwd, current_model, probe_timeout)
    if models:
        return models

    # Fallback for coco — try CocoModelManager again (probe inside it may have
    # populated cache concurrently) before degrading to DEFAULT_MODELS.
    cached = _coco_models_from_manager(current_model)
    if cached:
        return cached
    try:
        from ..coco_model.manager import DEFAULT_MODELS

        logger.warning(
            "[ACP] coco ACP probe returned no models, falling back to %d static DEFAULT_MODELS",
            len(DEFAULT_MODELS),
        )
        target_default = _coco_target_default(current_model)
        return [
            ACPModelOption(
                name=m.name,
                description=m.description,
                is_default=bool(
                    (target_default and m.name == target_default)
                    or getattr(m, "is_default", False)
                ),
            )
            for m in DEFAULT_MODELS
            if getattr(m, "name", "")
        ]
    except Exception:
        logger.warning("[ACP] coco model fallback failed", exc_info=True)

    fallback = _local_fallback_models(tool_name, current_model)
    if fallback:
        return fallback

    return []


def kickoff_acp_model_preheat(
    tool_names: list[str],
    cwd: str,
) -> threading.Thread | None:
    """Best-effort background preheat for ACP model capability caches."""
    normalized_tools: list[str] = []
    seen: set[str] = set()
    for raw_tool_name in tool_names:
        tool_name = str(raw_tool_name or "").strip()
        if not tool_name or tool_name in seen:
            continue
        seen.add(tool_name)
        normalized_tools.append(tool_name)

    if not normalized_tools:
        return None

    def _preheat() -> None:
        for tool_name in normalized_tools:
            started_at = _time.monotonic()
            try:
                models = fetch_acp_models(tool_name, cwd=cwd)
                count = len(models)
                outcome = "success" if models else "empty"
                logger.info(
                    "[ACP] model preheat tool=%s count=%d outcome=%s duration_ms=%.1f",
                    tool_name,
                    count,
                    outcome,
                    (_time.monotonic() - started_at) * 1000,
                )
            except Exception:
                logger.info(
                    "[ACP] model preheat tool=%s count=0 outcome=failed duration_ms=%.1f",
                    tool_name,
                    (_time.monotonic() - started_at) * 1000,
                    exc_info=True,
                )

    thread = threading.Thread(
        target=_preheat,
        name="acp-model-preheat",
        daemon=True,
    )
    thread.start()
    return thread


def _coco_target_default(current_model: Optional[str]) -> str:
    """Resolve the model name to mark as default for Coco rendering."""
    try:
        from ..coco_model import get_coco_model_manager

        configured_current = None
        try:
            configured_current = get_coco_model_manager().get_current_model()
        except Exception:
            logger.debug("[ACP] coco current model lookup failed", exc_info=True)
        return str(current_model or configured_current or "").strip()
    except Exception:
        return str(current_model or "").strip()


def _coco_models_from_manager(current_model: Optional[str]) -> list[ACPModelOption]:
    """Read Coco models from ``CocoModelManager`` (cache + ACP probe + static).

    Returns the same list ``/coco_status`` and the agent bootstrap rely on,
    so model selection cards stay in sync with the rest of the system.
    Returns an empty list when CocoModelManager has not yet populated and
    the caller should still attempt a fresh probe.
    """
    try:
        from ..coco_model import get_coco_model_manager
        from ..coco_model.manager import DEFAULT_MODELS

        manager = get_coco_model_manager()
        result = manager.get_models()
        models = list(result.models or [])
        if not models:
            return []
        # If manager only had time to return the static defaults (probe failed
        # or never ran), let the caller try a fresh ACP probe; we can come back
        # to manager later if probe also fails.
        default_names = {m.name for m in DEFAULT_MODELS}
        unique_names = {m.name for m in models}
        if unique_names == default_names:
            return []
        target_default = _coco_target_default(current_model)
        out: list[ACPModelOption] = []
        for m in models:
            name = str(getattr(m, "name", "") or "").strip()
            if not name:
                continue
            description = str(getattr(m, "description", "") or name)
            is_default = bool(
                (target_default and name == target_default)
                or getattr(m, "is_default", False)
            )
            out.append(
                ACPModelOption(name=name, description=description, is_default=is_default)
            )
        return out
    except Exception:
        logger.debug("[ACP] coco manager lookup failed", exc_info=True)
        return []


def _resolve_acp_model_probe_timeout(probe_timeout: Optional[float] = None) -> float:
    """Resolve the probe timeout for fetch_acp_models.

    Prefer ``acp_model_probe_timeout`` (designed for full model-list probing),
    fall back to the legacy ``acp_healthcheck_timeout`` for backwards-compat
    when the dedicated setting is unset.
    """
    if probe_timeout is not None:
        return max(0.1, float(probe_timeout))
    try:
        settings = get_settings()
        configured = float(
            getattr(settings, "acp_model_probe_timeout", None)
            or getattr(settings, "acp_healthcheck_timeout", 2.0)
            or 2.0
        )
    except Exception:
        configured = 6.0
    return max(0.1, configured)


class SessionKeyCodec:
    """`session_key` 编解码协作者。

    设计目标：
    - 将会话路由使用的 `session_key` 字符串协议（chat/project/thread）集中到
      单一位置，避免在各处手写字符串拼接或拆分逻辑；
    - 保持与现有 :class:`ACPSessionManager` 中 `_session_key` /
      `_parse_session_key` 语义等价，包括默认项目占位符与旧格式兼容策略；
    - 提供面向调用方的显式类型签名，便于在测试中做 roundtrip 与异常路径
      覆盖，同时为后续迁移 Lint 提供目标入口。

    注意：
    - 本类仅负责「字符串协议 ↔ 结构化三元组(chat_id, project_id, thread_id)」
      的转换，不做持久化、日志打印或安全校验；
    - 默认项目的占位符常量应与 ACPSessionManager 中使用的值保持一致，
      后续在完成迁移后会以本类为 SSOT。
    """

    #: 默认 project 段占位符；必须与 ACPSessionManager 中的 `_DEFAULT_PROJECT`
    #: 保持一致，以确保旧 key 仍能被正确解析。
    DEFAULT_PROJECT_PLACEHOLDER = "_default_"

    @classmethod
    def encode(
        cls,
        chat_id: str,
        project_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """根据 chat/project/thread 构造用于内部路由的 `session_key`。

        约定：
        - `chat_id` 始终位于首段且不可为空字符串；
        - 第二段为 project 占位：有显式 project 时使用其字符串形式；否则使用
          `_default_` 占位，调用方通过 `project_id is None` 语义区分；
        - 存在 `thread_id` 时，在 project 段之后追加 `":t:"` 前缀形成
          `chat:project:t:thread_id`，以支持同 chat+project 下多线程隔离；
        - 所有输入均通过 `f"{value}"` 做字符串化，不做字符合法性归一化。
        """

        # 与现有实现保持一致的字符串化策略
        base = f"{chat_id}:{project_id}" if project_id else f"{chat_id}:{cls.DEFAULT_PROJECT_PLACEHOLDER}"
        if thread_id:
            return f"{base}:t:{thread_id}"
        return base

    @classmethod
    def decode(cls, key: str) -> tuple[str, Optional[str], Optional[str]]:
        """将 `session_key` 解析为 `(chat_id, project_id, thread_id)`。

        兼容约束：
        - 对称性：与 :meth:`encode` 的编码协议保持对称；
        - 宽进严出：对于历史/异常 key 保持鲁棒而不抛异常，极端场景下返回
          `("", None, None)` 或 `(key, None, None)` 保留可追踪信息；
        - 线程维度采用 `":t:"` 前缀的标准编码格式。
        """

        try:
            s = str(key or "")
        except Exception:
            # 极端兜底：保证返回可打印 chat_id
            return "", None, None

        if not s:
            return "", None, None

        try:
            parts = s.split(":")
            if not parts:
                return "", None, None

            chat_id = parts[0] or ""

            project_id: Optional[str] = None
            if len(parts) >= 2:
                raw_project = parts[1] or ""
                if raw_project and raw_project != cls.DEFAULT_PROJECT_PLACEHOLDER:
                    project_id = raw_project

            thread_id: Optional[str] = None
            # 标准编码：chat:project:t:thread_id
            if len(parts) >= 4 and parts[2] == "t":
                thread_id = parts[3] or ""

            return chat_id, project_id, thread_id
        except Exception:
            # 解析失败时，保留原始 key 作为 chat_id，避免完全丢失上下文
            return s, None, None


def _inject_claude_1m_variants(items: list[ACPModelOption]) -> list[ACPModelOption]:
    """For each Anthropic model that supports the 1M-context beta, append a
    sibling ``ACPModelOption`` carrying the ``[1m]`` suffix.

    The original entry is preserved verbatim.  Variants inherit the original
    ``description`` (rendered behind a 🚀 emoji + 1M tag) and never claim
    ``is_default=True`` — selecting them must be an explicit user choice
    because >200K tokens are billed at the higher tier.
    """
    from .claude_capabilities import (
        is_1m_variant,
        model_supports_1m,
        with_1m_suffix,
    )

    seen = {opt.name for opt in items}
    extras: list[ACPModelOption] = []
    for opt in list(items):
        if is_1m_variant(opt.name):
            continue
        if not model_supports_1m(opt.name):
            continue
        variant_name = with_1m_suffix(opt.name)
        if variant_name in seen:
            continue
        seen.add(variant_name)
        base_desc = (opt.description or opt.name).strip()
        extras.append(
            ACPModelOption(
                name=variant_name,
                description=f"🚀 1M 上下文（{base_desc}，>200K tokens 计费上调）",
                is_default=False,
                supports_1m=True,
            )
        )
    if extras:
        items.extend(extras)
    return items


async def probe_acp_models(
    tool_name: str, cwd: Optional[str], current_model: Optional[str] = None
) -> list[ACPModelOption]:
    """Asynchronously probe available models from an ACP provider."""
    get_providers()
    provider = tool_registry.get_provider(tool_name)
    if not provider:
        return []

    cmd, args = provider.get_serve_command(None)

    from ..utils.env import build_clean_env

    env = build_clean_env()
    client = GhostAPClient(on_event=lambda _ev: None, auto_approve=True)

    try:
        async with spawn_agent_process(
            client, cmd, *args, env=env, cwd=(cwd or str(Path.cwd()))
        ) as (conn, _proc):
            await conn.initialize(protocol_version=1)
            resp = await conn.new_session(cwd=(cwd or str(Path.cwd())))
            if tool_name == "traex":
                return await _extract_traex_model_capabilities(
                    conn,
                    resp,
                    current_model=current_model,
                )
            if tool_name == "codex":
                return await _extract_codex_model_capabilities(
                    conn,
                    resp,
                    current_model=current_model,
                )

            models_state = getattr(resp, "models", None)
            available = list(getattr(models_state, "available_models", []) or [])
            raw_current_id = str(
                getattr(models_state, "current_model_id", "")
                or getattr(models_state, "currentModelId", "")
            ).strip()
            current_id = raw_current_id.split("/")[0] if "/" in raw_current_id else raw_current_id
            target_default = str((current_model or current_id or "")).strip()

            items = []
            seen = set()
            for item in available:
                model_id = str(
                    getattr(item, "model_id", "")
                    or getattr(item, "modelId", "")
                    or getattr(item, "name", "")
                ).strip()
                if not model_id or model_id in seen:
                    continue
                seen.add(model_id)
                description = str(
                    getattr(item, "description", "")
                    or getattr(item, "name", "")
                    or model_id
                ).strip()
                items.append(
                    ACPModelOption(
                        name=model_id,
                        description=description,
                        is_default=(model_id == target_default),
                    )
                )

            if not items:
                items = _extract_models_from_config_options(resp, target_default)

            if tool_name == "claude":
                items = _inject_claude_1m_variants(items)

            return items
    except Exception as e:
        from ..utils.errors import get_error_detail

        logger.info(
            "[ACP] fetch models failed: tool=%s err=%s", tool_name, get_error_detail(e)
        )
        return []


def _config_option_roots(resp: object) -> list[object]:
    return [
        getattr(option, "root", option)
        for option in (getattr(resp, "config_options", None) or [])
    ]


def _find_config_option(
    resp: object,
    *,
    option_id: str,
    category: Optional[str] = None,
) -> Optional[object]:
    for root in _config_option_roots(resp):
        root_id = str(getattr(root, "id", "") or "").strip()
        root_category = str(getattr(root, "category", "") or "").strip()
        if root_id == option_id or (category and root_category == category):
            return root
    return None


def _iter_config_select_options(root: Optional[object]) -> list[object]:
    if root is None:
        return []
    flattened: list[object] = []
    for option in (getattr(root, "options", None) or []):
        nested = getattr(option, "options", None)
        if nested is not None:
            flattened.extend(list(nested or []))
        else:
            flattened.append(option)
    return flattened


async def _extract_traex_model_capabilities(
    conn: object,
    resp: object,
    *,
    current_model: Optional[str],
    metadata_path: Optional[Path] = None,
) -> list[ACPModelOption]:
    from .model_selection import CODEX_REASONING_EFFORTS
    from .traex_selection import (
        TraexProfileMetadata,
        compose_traex_model_selection,
        load_traex_model_metadata,
        split_traex_model_selection,
    )

    model_root = _find_config_option(resp, option_id="model", category="model")
    session_id = str(getattr(resp, "session_id", "") or "").strip()
    set_config_option = getattr(conn, "set_config_option", None)
    if model_root is None or not session_id or not callable(set_config_option):
        return []

    metadata_by_name = {}
    for metadata in load_traex_model_metadata(metadata_path):
        metadata_by_name[metadata.config_name] = metadata
        metadata_by_name[metadata.slug] = metadata

    requested_model, _profile, _effort = split_traex_model_selection(
        current_model
    )
    adapter_current = str(
        getattr(model_root, "current_value", "") or ""
    ).strip()
    target_default = requested_model or adapter_current
    models: list[ACPModelOption] = []
    seen: set[str] = set()

    for option in _iter_config_select_options(model_root):
        model_id = str(getattr(option, "value", "") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        metadata = metadata_by_name.get(model_id)
        profile_options: tuple[TraexProfileMetadata, ...]
        if metadata is not None:
            profile_options = metadata.profiles
        else:
            profile_options = (
                TraexProfileMetadata(
                    profile="standard",
                    backend_model_value=model_id,
                ),
            )

        variants: list[ACPModelVariantOption] = []
        for profile in profile_options:
            try:
                changed = await set_config_option(
                    session_id=session_id,
                    config_id="model",
                    value=profile.backend_model_value,
                )
            except Exception:
                logger.warning(
                    "[ACP] traex capability probe rejected model=%s profile=%s",
                    model_id,
                    profile.profile,
                    exc_info=True,
                )
                continue

            reasoning_root = _find_config_option(
                changed,
                option_id="reasoning_effort",
                category="thought_level",
            )
            live_efforts = tuple(
                value
                for value in (
                    str(getattr(item, "value", "") or "").strip().lower()
                    for item in _iter_config_select_options(reasoning_root)
                )
                if value in CODEX_REASONING_EFFORTS
            )
            if profile.reasoning_efforts:
                efforts = tuple(
                    value
                    for value in profile.reasoning_efforts
                    if value in live_efforts
                )
            else:
                efforts = live_efforts

            adapter_effort = str(
                getattr(reasoning_root, "current_value", "") or ""
            ).strip().lower()
            default_effort = (
                adapter_effort
                if adapter_effort in efforts
                else (
                    profile.default_effort
                    if profile.default_effort in efforts
                    else (efforts[0] if efforts else None)
                )
            )
            if efforts:
                for effort in efforts:
                    selection_name = compose_traex_model_selection(
                        model_id,
                        profile.profile,
                        effort,
                    )
                    variants.append(
                        ACPModelVariantOption(
                            name=selection_name,
                            profile=profile.profile,
                            effort=effort,
                            display_name=(
                                f"{getattr(option, 'name', None) or model_id}"
                                f" · {profile.profile} · {effort}"
                            ),
                            is_variant_default=(effort == default_effort),
                        )
                    )
            else:
                selection_name = compose_traex_model_selection(
                    model_id,
                    profile.profile,
                    None,
                )
                variants.append(
                    ACPModelVariantOption(
                        name=selection_name,
                        profile=profile.profile,
                        display_name=(
                            f"{getattr(option, 'name', None) or model_id}"
                            f" · {profile.profile}"
                        ),
                        is_variant_default=True,
                    )
                )

        if not variants:
            continue
        models.append(
            ACPModelOption(
                name=model_id,
                description=str(
                    getattr(option, "name", "")
                    or getattr(option, "description", "")
                    or model_id
                ).strip(),
                is_default=(
                    target_default in {model_id, getattr(metadata, "slug", "")}
                ),
                selection_variants=tuple(variants),
            )
        )

    logger.debug(
        "[ACP] extracted %d Traex models with profile capabilities",
        len(models),
    )
    return models


async def _extract_codex_model_capabilities(
    conn: object,
    resp: object,
    *,
    current_model: Optional[str],
) -> list[ACPModelOption]:
    """Read the official Codex adapter's model-specific Effort matrix."""
    from .model_selection import (
        CODEX_REASONING_EFFORTS,
        split_codex_model_selection,
    )

    model_root = _find_config_option(resp, option_id="model", category="model")
    reasoning_root = _find_config_option(
        resp,
        option_id="reasoning_effort",
        category="thought_level",
    )
    if model_root is None or reasoning_root is None:
        return []

    selected_model, _selected_effort = split_codex_model_selection(current_model)
    adapter_current = str(getattr(model_root, "current_value", "") or "").strip()
    adapter_effort = str(
        getattr(reasoning_root, "current_value", "") or ""
    ).strip()
    target_default = selected_model or adapter_current
    session_id = str(getattr(resp, "session_id", "") or "").strip()
    if not session_id or not adapter_current or not adapter_effort:
        return []

    items: list[ACPModelOption] = []
    seen: set[str] = set()
    set_config_option = getattr(conn, "set_config_option", None)
    if not callable(set_config_option):
        return []

    for option in _iter_config_select_options(model_root):
        model_id = str(getattr(option, "value", "") or "").strip()
        if not model_id or model_id in seen:
            continue
        try:
            await set_config_option(
                session_id=session_id,
                config_id="model",
                value=adapter_current,
            )
            await set_config_option(
                session_id=session_id,
                config_id="reasoning_effort",
                value=adapter_effort,
            )
            changed = await set_config_option(
                session_id=session_id,
                config_id="model",
                value=model_id,
            )
        except Exception:
            logger.warning(
                "[ACP] codex capability probe rejected model=%s",
                model_id,
                exc_info=True,
            )
            continue

        current_reasoning = _find_config_option(
            changed,
            option_id="reasoning_effort",
            category="thought_level",
        )
        efforts = tuple(
            value
            for value in (
                str(getattr(effort, "value", "") or "").strip()
                for effort in _iter_config_select_options(current_reasoning)
            )
            if value in CODEX_REASONING_EFFORTS
        )
        raw_efforts = tuple(
            str(getattr(effort, "value", "") or "").strip()
            for effort in _iter_config_select_options(current_reasoning)
        )
        unknown_efforts = sorted(
            {value for value in raw_efforts if value}
            - CODEX_REASONING_EFFORTS
        )
        if unknown_efforts:
            logger.warning(
                "[ACP] ignoring unsupported Codex reasoning efforts: model=%s efforts=%s",
                model_id,
                unknown_efforts,
            )
        adapted_effort = str(
            getattr(current_reasoning, "current_value", "") or ""
        ).strip() or None
        if not efforts or adapted_effort not in efforts:
            logger.warning(
                "[ACP] codex capability probe returned no valid efforts: model=%s",
                model_id,
            )
            continue

        seen.add(model_id)
        items.append(
            ACPModelOption(
                name=model_id,
                description=str(
                    getattr(option, "name", "")
                    or getattr(option, "description", "")
                    or model_id
                ).strip(),
                is_default=(model_id == target_default),
                reasoning_efforts=efforts,
                adapted_reasoning_effort=adapted_effort,
            )
        )

    logger.debug(
        "[ACP] extracted %d Codex models with reasoning effort capabilities",
        len(items),
    )
    return items


def _extract_models_from_config_options(
    resp: object, target_default: str
) -> list[ACPModelOption]:
    """Extract model list from config_options when available_models is empty.

    Some ACP providers (e.g. traex) return models only via the
    ``config_options`` field with ``category='model'`` instead of via the
    ``models.available_models`` array.
    """
    config_options = getattr(resp, "config_options", None)
    if not config_options:
        return []

    for opt in config_options:
        root = getattr(opt, "root", opt)
        category = getattr(root, "category", None) or ""
        if category != "model":
            continue

        current_value = str(getattr(root, "current_value", "") or "").strip()
        effective_default = target_default or current_value
        options = getattr(root, "options", None) or []

        items: list[ACPModelOption] = []
        seen: set[str] = set()
        for option in options:
            if hasattr(option, "options"):
                group_options = getattr(option, "options", []) or []
                for go in group_options:
                    _add_config_option_model(go, effective_default, seen, items)
            else:
                _add_config_option_model(option, effective_default, seen, items)

        if items:
            logger.debug(
                "[ACP] extracted %d models from config_options (category=model)",
                len(items),
            )
            return items

    return []


def _add_config_option_model(
    option: object,
    effective_default: str,
    seen: set[str],
    items: list[ACPModelOption],
) -> None:
    """Add a single model entry from a SessionConfigSelectOption."""
    value = str(getattr(option, "value", "") or "").strip()
    if not value or value in seen:
        return
    seen.add(value)
    name_label = str(getattr(option, "name", "") or value).strip()
    items.append(
        ACPModelOption(
            name=value,
            description=name_label,
            is_default=(value == effective_default),
        )
    )
