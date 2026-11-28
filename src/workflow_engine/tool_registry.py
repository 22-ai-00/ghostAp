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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

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
_cached_capabilities: dict[str, tuple[float, int, dict[str, "ToolCapability"]]] = {}

EFFORT_CAPABLE_TOOLS: frozenset[str] = frozenset(
    {"codex", "codex_w", "claude", "claude_w", "agy", "agy_b", "traex", "dsh"}
)
MODEL_ONLY_TOOLS: frozenset[str] = frozenset({"coco", "aiden", "gemini", "grok"})
_CLI_PSEUDO_MODEL_TOOLS: frozenset[str] = frozenset({"claude", "claude_w", "codex_w"})


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """Immutable capability snapshot for one Workflow programming tool backend."""

    tool_name: str
    display_name: str
    available: bool
    supports_model: bool
    supports_effort: bool
    models: tuple[str, ...] = ()
    efforts_by_model: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    default_effort: Optional[str] = None
    profiles_by_model: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    description: str = ""
    authoritative_models: bool = False

    def __post_init__(self) -> None:
        tool_name = str(self.tool_name or "").strip().lower()
        display_name = str(self.display_name or tool_name).strip() or tool_name
        models = tuple(
            dict.fromkeys(
                str(model).strip()
                for model in (self.models or ())
                if str(model or "").strip()
            )
        )
        frozen_efforts = MappingProxyType(
            {
                str(model).strip(): tuple(
                    dict.fromkeys(
                        str(item).strip().lower()
                        for item in (efforts or ())
                        if str(item or "").strip()
                    )
                )
                for model, efforts in (self.efforts_by_model or {}).items()
                if str(model or "").strip()
            }
        )
        frozen_profiles = MappingProxyType(
            {
                str(model).strip(): tuple(
                    dict.fromkeys(
                        str(item).strip().lower()
                        for item in (profiles or ())
                        if str(item or "").strip()
                    )
                )
                for model, profiles in (self.profiles_by_model or {}).items()
                if str(model or "").strip()
            }
        )
        default_effort = str(self.default_effort or "").strip().lower() or None
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "supports_model", bool(self.supports_model))
        object.__setattr__(self, "supports_effort", bool(self.supports_effort))
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "efforts_by_model", frozen_efforts)
        object.__setattr__(self, "default_effort", default_effort)
        object.__setattr__(self, "profiles_by_model", frozen_profiles)
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "authoritative_models", bool(self.authoritative_models))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "display_name": self.display_name,
            "available": self.available,
            "supports_model": self.supports_model,
            "supports_effort": self.supports_effort,
            "models": list(self.models),
            "efforts_by_model": {k: list(v) for k, v in self.efforts_by_model.items()},
            "default_effort": self.default_effort,
            "profiles_by_model": {k: list(v) for k, v in self.profiles_by_model.items()},
            "description": self.description,
            "authoritative_models": self.authoritative_models,
        }

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
    "agy": "Google Antigravity 原生 ACP 编程",
    "agy_b": "Google Antigravity 原生 ACP 编程 (AGY-B)",
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
    """Force the next get_available_tools()/get_tool_capabilities() call to re-discover."""
    global _cached_selectable_at, _cached_selectable_tools, _cached_tools, _cached_at
    with _cache_lock:
        _cached_tools = None
        _cached_at = 0.0
        _cached_selectable_tools = None
        _cached_selectable_at = 0.0
        _cached_capabilities.clear()


def _acp_cache_version() -> int:
    try:
        from ..acp.helper import get_acp_model_cache_version

        return int(get_acp_model_cache_version())
    except Exception:
        return 0


def get_tool_capabilities(
    force_refresh: bool = False,
    *,
    cwd: str | None = None,
) -> dict[str, ToolCapability]:
    """Return cached non-blocking capabilities for all Workflow backends.

    Returns a fresh dict mapping ``tool_name -> ToolCapability``. Never runs a
    synchronous ACP probe on cache miss. Invalidates immediately when the
    underlying ACP model probe cache version advances.
    """
    cache_key = str(cwd or "")
    acp_version = _acp_cache_version()
    now = time.monotonic()
    if not force_refresh:
        with _cache_lock:
            entry = _cached_capabilities.get(cache_key)
            if entry is not None and (now - entry[0]) < _CACHE_TTL_S and entry[1] == acp_version:
                return dict(entry[2])

    with _cache_lock:
        acp_version = _acp_cache_version()
        if not force_refresh:
            entry = _cached_capabilities.get(cache_key)
            if (
                entry is not None
                and (time.monotonic() - entry[0]) < _CACHE_TTL_S
                and entry[1] == acp_version
            ):
                return dict(entry[2])

        capabilities = _discover_tool_capabilities(cwd=cwd)
        # Stamp with pre-discovery acp_version so concurrent _store() updates
        # during _discover_tool_capabilities() invalidate on the next read.
        _cached_capabilities[cache_key] = (
            time.monotonic(),
            acp_version,
            capabilities,
        )
        return dict(capabilities)


def _extract_option_capabilities(
    tool_name: str,
    options: list[Any],
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    Optional[str],
]:
    models: list[str] = []
    efforts_by_model: dict[str, tuple[str, ...]] = {}
    profiles_by_model: dict[str, tuple[str, ...]] = {}
    default_effort: Optional[str] = None

    for option in options:
        model_name = str(getattr(option, "name", "") or "").strip()
        if not model_name:
            continue
        if model_name not in models:
            models.append(model_name)

        raw_efforts = [
            str(item).strip().lower()
            for item in (getattr(option, "reasoning_efforts", ()) or ())
            if str(item or "").strip()
        ]
        variants = getattr(option, "selection_variants", ()) or ()
        variant_efforts: list[str] = []
        variant_profiles: list[str] = []
        for variant in variants:
            v_effort = str(getattr(variant, "effort", "") or "").strip().lower()
            v_profile = str(getattr(variant, "profile", "") or "").strip().lower()
            if v_profile and (tool_name not in {"agy", "agy_b"} or v_profile != "default"):
                if v_profile not in variant_profiles:
                    variant_profiles.append(v_profile)
            if v_effort:
                if tool_name in _CLI_PSEUDO_MODEL_TOOLS and v_effort == "default":
                    continue
                if v_effort not in variant_efforts:
                    variant_efforts.append(v_effort)
            if getattr(variant, "is_default", False) and default_effort is None:
                if v_effort and not (tool_name in _CLI_PSEUDO_MODEL_TOOLS and v_effort == "default"):
                    default_effort = v_effort

        merged_efforts = tuple(dict.fromkeys([*raw_efforts, *variant_efforts]))
        if merged_efforts:
            efforts_by_model[model_name] = merged_efforts
        if variant_profiles:
            profiles_by_model[model_name] = tuple(variant_profiles)

        opt_default_effort = str(
            getattr(option, "default_reasoning_effort", "") or ""
        ).strip().lower()
        if opt_default_effort and getattr(option, "is_default", False):
            default_effort = opt_default_effort
        elif opt_default_effort and default_effort is None:
            default_effort = opt_default_effort

    return tuple(models), efforts_by_model, profiles_by_model, default_effort


def _discover_tool_capabilities(*, cwd: str | None = None) -> dict[str, ToolCapability]:
    from ..acp.helper import get_cached_acp_models, is_programming_tool_available
    from ..acp.providers import get_providers
    from ..acp.traex_selection import load_traex_model_metadata
    from ..agent_session.backend_resolver import cli_command_for_agent, is_cli_backend
    from ..mode.backends import ALL_BACKEND_SPECS

    try:
        providers = get_providers()
    except Exception as exc:
        logger.debug("get_providers() failed during capability discovery: %s", exc)
        providers = {}

    desc_map: dict[str, str] = {}
    try:
        from ..utils.text import get_acp_result_header_text

        headers = get_acp_result_header_text()
        for spec in ALL_BACKEND_SPECS:
            key = f"tool_desc_{spec.mode_key}"
            if key in headers:
                desc_map[spec.mode_key] = headers[key]
    except Exception:
        pass

    capabilities: dict[str, ToolCapability] = {}
    for spec in ALL_BACKEND_SPECS:
        tool_name = spec.mode_key
        available = False
        if is_cli_backend(tool_name):
            cli_cmd = cli_command_for_agent(tool_name)
            available = shutil.which(cli_cmd) is not None
        else:
            if shutil.which(spec.cli_name) is not None or (
                spec.cli_name != tool_name and shutil.which(tool_name) is not None
            ):
                available = True
                try:
                    is_programming_tool_available(
                        tool_name,
                        allow_sync_probe=False,
                        trigger_async_probe=True,
                    )
                except Exception:
                    logger.debug("Async ACP availability warmup failed for %s", tool_name, exc_info=True)
            else:
                provider = providers.get(tool_name)
                get_fallback = getattr(provider, "get_fallback_command", None) if provider else None
                try:
                    available = bool(callable(get_fallback) and get_fallback())
                except Exception:
                    available = False
                if not available:
                    try:
                        available = bool(
                            is_programming_tool_available(
                                tool_name,
                                allow_sync_probe=False,
                                trigger_async_probe=True,
                            )
                        )
                    except Exception:
                        available = False

        supports_effort = tool_name in EFFORT_CAPABLE_TOOLS
        models: tuple[str, ...] = ()
        efforts_by_model: dict[str, tuple[str, ...]] = {}
        profiles_by_model: dict[str, tuple[str, ...]] = {}
        default_effort: Optional[str] = None

        try:
            cached_options = get_cached_acp_models(tool_name, cwd)
        except Exception:
            cached_options = []

        authoritative_models = False
        if cached_options:
            models, efforts_by_model, profiles_by_model, default_effort = (
                _extract_option_capabilities(tool_name, cached_options)
            )
            authoritative_models = bool(models)
        elif tool_name == "traex":
            try:
                traex_meta = load_traex_model_metadata()
            except Exception:
                traex_meta = ()
            if traex_meta:
                meta_models: list[str] = []
                for item in traex_meta:
                    meta_models.append(item.config_name)
                    prof_names = tuple(p.profile for p in item.profiles if p.profile)
                    if prof_names:
                        profiles_by_model[item.config_name] = prof_names
                    eff_list: list[str] = []
                    for prof in item.profiles:
                        for eff in prof.reasoning_efforts:
                            if eff not in eff_list:
                                eff_list.append(eff)
                        if (
                            default_effort is None
                            and prof.profile == "standard"
                            and prof.default_effort
                        ):
                            default_effort = prof.default_effort
                    if eff_list:
                        efforts_by_model[item.config_name] = tuple(eff_list)
                models = tuple(dict.fromkeys(meta_models))
                authoritative_models = bool(models)

        if not supports_effort:
            efforts_by_model = {}
            default_effort = None

        description = (
            desc_map.get(tool_name)
            or _FALLBACK_DESCRIPTIONS.get(tool_name)
            or spec.display_name
        )
        capabilities[tool_name] = ToolCapability(
            tool_name=tool_name,
            display_name=spec.display_name,
            available=available,
            supports_model=True,
            supports_effort=supports_effort,
            models=models,
            efforts_by_model=efforts_by_model,
            default_effort=default_effort,
            profiles_by_model=profiles_by_model,
            description=description,
            authoritative_models=authoritative_models,
        )

    return capabilities


def _is_tool_available_against_filter(
    tool: str,
    cap: ToolCapability,
    available_tools: Mapping[str, Any] | Iterable[str] | None,
) -> bool:
    if available_tools is None:
        return cap.available
    if isinstance(available_tools, Mapping):
        if tool not in available_tools:
            return False
        entry = available_tools[tool]
        if isinstance(entry, ToolCapability):
            return entry.available
        return bool(entry) if entry is not None else True
    allowed = {str(item or "").strip().lower() for item in available_tools}
    return tool in allowed


def validate_agent_binding_spec(
    tool_name: str,
    model_name: str | None = None,
    profile: str | None = None,
    effort: str | None = None,
    available_tools: Mapping[str, Any] | Iterable[str] | None = None,
    *,
    cwd: str | None = None,
    capabilities: Mapping[str, ToolCapability] | None = None,
) -> tuple[bool, Optional[str]]:
    """Validate a candidate ``(tool_name, model_name, profile, effort)`` binding."""
    tool = str(tool_name or "").strip().lower()
    if not tool:
        return False, "工具名称不能为空"

    caps = capabilities if capabilities is not None else get_tool_capabilities(cwd=cwd)
    cap = caps.get(tool)
    if cap is None:
        return False, f"未知工具: {tool_name}（可选: {', '.join(sorted(caps))}）"

    if not _is_tool_available_against_filter(tool, cap, available_tools):
        return False, f"工具 {tool} 当前不可用，请检查本地安装或环境配置"

    raw_model = str(model_name or "").strip() or None
    raw_profile = str(profile or "").strip().lower() or None
    raw_effort = str(effort or "").strip().lower() or None
    if raw_effort == "default":
        raw_effort = None

    if not cap.supports_effort:
        if raw_effort is not None:
            return False, f"工具 {tool} 不支持设置 effort（收到 effort={raw_effort}）"
        if raw_profile is not None:
            return False, f"工具 {tool} 不支持设置 profile（收到 profile={raw_profile}）"
        if (
            raw_model is not None
            and raw_model.lower() != "default"
            and cap.models
            and raw_model not in cap.models
        ):
            return (
                False,
                f"工具 {tool} 不支持模型: {raw_model}（可用: {', '.join(cap.models[:8])}）",
            )
        return True, None

    from ..acp.agy_selection import (
        AGY_ORCHESTRATOR_DEFAULT_PROFILE,
        AGY_REASONING_EFFORTS,
        is_agy_orchestrator_model,
        is_valid_agy_executor_model,
        is_valid_agy_planner_model,
        parse_agy_model_selection,
        split_agy_model_selection,
    )
    from ..acp.claude_selection import (
        CLAUDE_DEFAULT_EFFORT_TOKEN,
        CLAUDE_REASONING_EFFORTS,
        split_claude_model_selection,
    )
    from ..acp.dsh_selection import (
        DSH_DEFAULT_REASONING,
        decode_dsh_model_value,
        decode_dsh_reasoning_value,
        split_dsh_model_selection,
    )
    from ..acp.model_selection import (
        CODEX_DEFAULT_EFFORT_TOKEN,
        CODEX_REASONING_EFFORTS,
        split_codex_model_selection,
    )
    from ..acp.traex_selection import (
        load_traex_model_metadata,
        split_traex_model_selection,
    )

    def _find_traex_meta(target: str) -> Any:
        try:
            items = load_traex_model_metadata()
        except Exception:
            return None
        for item in items or ():
            if target in {item.config_name, item.slug}:
                return item
        return None

    base_model: Optional[str] = raw_model
    emb_profile: Optional[str] = None
    emb_effort: Optional[str] = None
    agy_planner_model: Optional[str] = None
    agy_executor_model: Optional[str] = None
    traex_meta = None

    if tool in {"claude", "claude_w"}:
        if raw_model is not None:
            base_model, emb_effort = split_claude_model_selection(raw_model)
    elif tool in {"codex", "codex_w"}:
        if raw_model is not None:
            base_model, emb_effort = split_codex_model_selection(raw_model)
    elif tool == "traex":
        if raw_model is not None:
            parts = [part for part in raw_model.split("/") if part]
            emb_base, parsed_profile, parsed_effort = split_traex_model_selection(raw_model)
            base_model = emb_base or None
            emb_effort = parsed_effort
            if len(parts) >= 3 or (len(parts) >= 2 and parts[-1].lower() in {"standard", "max"}):
                emb_profile = parsed_profile
    elif tool in {"agy", "agy_b"}:
        if raw_model is not None:
            if is_agy_orchestrator_model(raw_model):
                try:
                    parsed = parse_agy_model_selection(raw_model)
                except ValueError:
                    return False, f"无效的 AGY Orchestrator 模型或 effort 配置: {raw_model}"
                base_model = parsed.model_id
                emb_profile = parsed.planner_effort
                emb_effort = parsed.executor_effort
                agy_planner_model = parsed.planner_model
                agy_executor_model = parsed.executor_model
                if agy_planner_model and not is_valid_agy_planner_model(agy_planner_model):
                    return False, f"AGY Orchestrator 不支持的 Planner 模型: {agy_planner_model}"
                if agy_executor_model and not is_valid_agy_executor_model(agy_executor_model):
                    return False, f"AGY Orchestrator 不支持的 Executor 模型: {agy_executor_model}"
            else:
                try:
                    base_model, emb_effort = split_agy_model_selection(raw_model)
                except ValueError:
                    return False, f"无效的 AGY 模型或 effort 配置: {raw_model}"
    elif tool == "dsh":
        if raw_model is not None:
            if raw_model.startswith("["):
                try:
                    model_json, reasoning_json = split_dsh_model_selection(raw_model)
                    provider, model_id = decode_dsh_model_value(model_json)
                except ValueError:
                    return False, f"无效的 DSH 模型格式: {raw_model}"
                base_model = f"{provider}/{model_id}"
                emb_effort = (
                    decode_dsh_reasoning_value(reasoning_json) if reasoning_json else None
                )
            else:
                if raw_model == "default":
                    base_model = "default"
                elif "/" not in raw_model or len([p for p in raw_model.split("/") if p]) != 2:
                    if not (cap.models and raw_model in cap.models):
                        return (
                            False,
                            f"无效的 DSH 模型格式（需要 provider/model）: {raw_model}",
                        )
                    base_model = raw_model
                else:
                    base_model = raw_model

    is_default_base = (
        base_model == "default"
        or (tool == "dsh" and base_model == "default/default")
    )

    effective_profile = raw_profile or emb_profile
    if effective_profile is not None:
        if tool == "traex":
            if effective_profile not in {"standard", "max"}:
                return (
                    False,
                    f"Traex 不支持的 profile: {effective_profile}（可选: standard, max）",
                )
            if base_model and not is_default_base and base_model in cap.profiles_by_model:
                allowed_profiles = cap.profiles_by_model[base_model]
                if allowed_profiles and effective_profile not in allowed_profiles:
                    return (
                        False,
                        f"Traex 模型 {base_model} 不支持 profile={effective_profile}"
                        f"（可用: {', '.join(allowed_profiles)}）",
                    )
        elif tool in {"agy", "agy_b"} and base_model and is_agy_orchestrator_model(base_model):
            plan_target = agy_planner_model or base_model
            plan_efforts = (
                cap.efforts_by_model.get(plan_target)
                or cap.efforts_by_model.get(base_model)
            )
            allowed_planner = (
                set(plan_efforts) | {AGY_ORCHESTRATOR_DEFAULT_PROFILE}
                if plan_efforts
                else set(AGY_REASONING_EFFORTS) | {AGY_ORCHESTRATOR_DEFAULT_PROFILE}
            )
            if effective_profile not in allowed_planner:
                return (
                    False,
                    f"AGY Orchestrator 不支持的 planner profile/effort: {effective_profile}",
                )
        else:
            return (
                False,
                f"工具 {tool}（模型 {base_model or 'default'}）不支持设置 profile={effective_profile}",
            )

    if (
        base_model is not None
        and not is_default_base
        and tool not in _CLI_PSEUDO_MODEL_TOOLS
        and cap.models
    ):
        if tool == "traex":
            traex_meta = _find_traex_meta(base_model)
            matched_models = {base_model}
            if traex_meta is not None:
                matched_models.update({traex_meta.config_name, traex_meta.slug})
            if not (matched_models & set(cap.models)):
                return (
                    False,
                    f"工具 {tool} 不支持模型: {base_model}（可用: {', '.join(cap.models[:8])}）",
                )
        elif base_model not in cap.models:
            return (
                False,
                f"工具 {tool} 不支持模型: {base_model}（可用: {', '.join(cap.models[:8])}）",
            )
        elif (
            tool in {"agy", "agy_b"}
            and is_agy_orchestrator_model(base_model)
            and cap.authoritative_models
        ):
            from ..acp.agy_selection import get_agy_orchestrator_spec

            orch_spec = get_agy_orchestrator_spec(base_model)
            if (
                agy_planner_model
                and (orch_spec is None or agy_planner_model != orch_spec.planner_model_id)
                and agy_planner_model not in cap.models
            ):
                return (
                    False,
                    f"工具 {tool} 不支持模型: {agy_planner_model}（可用: {', '.join(cap.models[:8])}）",
                )
            if (
                agy_executor_model
                and (orch_spec is None or agy_executor_model != orch_spec.executor_model_id)
                and agy_executor_model not in cap.models
            ):
                return (
                    False,
                    f"工具 {tool} 不支持模型: {agy_executor_model}（可用: {', '.join(cap.models[:8])}）",
                )

    effective_effort = raw_effort or emb_effort
    if effective_effort == "default" and tool != "dsh":
        effective_effort = None
    if effective_effort is not None:
        if tool in {"claude", "claude_w"}:
            allowed = set(CLAUDE_REASONING_EFFORTS) | {CLAUDE_DEFAULT_EFFORT_TOKEN}
            if effective_effort not in allowed:
                return (
                    False,
                    f"工具 {tool} 不支持的 effort: {effective_effort}"
                    f"（可选: {', '.join(CLAUDE_REASONING_EFFORTS)}）",
                )
        elif tool in {"codex", "codex_w"}:
            model_efforts = cap.efforts_by_model.get(base_model or "") if base_model else None
            if (
                tool == "codex"
                and cap.authoritative_models
                and base_model
                and not is_default_base
                and base_model in cap.models
                and not model_efforts
            ):
                return (
                    False,
                    f"Codex 模型 {base_model} 不支持设置 effort（收到 effort={effective_effort}）",
                )
            if model_efforts:
                allowed = set(model_efforts)
                if tool == "codex_w":
                    allowed.add(CODEX_DEFAULT_EFFORT_TOKEN)
                choices = ", ".join(model_efforts)
            else:
                allowed = set(CODEX_REASONING_EFFORTS)
                if tool == "codex_w":
                    allowed.add(CODEX_DEFAULT_EFFORT_TOKEN)
                choices = ", ".join(sorted(CODEX_REASONING_EFFORTS))
            if effective_effort not in allowed:
                return (
                    False,
                    f"工具 {tool} 不支持的 effort: {effective_effort}（可选: {choices}）",
                )
        elif tool in {"agy", "agy_b"}:
            if base_model and is_agy_orchestrator_model(base_model):
                exec_target = agy_executor_model or base_model
                model_efforts = (
                    cap.efforts_by_model.get(exec_target)
                    or cap.efforts_by_model.get(base_model)
                )
                allowed = (
                    set(model_efforts) | {AGY_ORCHESTRATOR_DEFAULT_PROFILE}
                    if model_efforts
                    else set(AGY_REASONING_EFFORTS) | {AGY_ORCHESTRATOR_DEFAULT_PROFILE}
                )
            else:
                model_efforts = cap.efforts_by_model.get(base_model or "") if base_model else None
                if (
                    cap.authoritative_models
                    and base_model
                    and not is_default_base
                    and base_model in cap.models
                    and not model_efforts
                ):
                    return (
                        False,
                        f"AGY 模型 {base_model} 不支持设置 effort（收到 effort={effective_effort}）",
                    )
                allowed = set(model_efforts) if model_efforts else set(AGY_REASONING_EFFORTS)
            if effective_effort not in allowed:
                return (
                    False,
                    f"工具 {tool} 不支持的 effort: {effective_effort}"
                    f"（可选: {', '.join(sorted(allowed))}）",
                )
        elif tool == "traex":
            traex_efforts: tuple[str, ...] | None = None
            if base_model and not is_default_base:
                traex_meta = traex_meta or _find_traex_meta(base_model)
                if traex_meta is not None:
                    target_profile = effective_profile or "standard"
                    prof_obj = next(
                        (item for item in traex_meta.profiles if item.profile == target_profile),
                        None,
                    )
                    if prof_obj is None:
                        return (
                            False,
                            f"Traex 模型 {base_model} 不支持 profile={target_profile}",
                        )
                    if not prof_obj.reasoning_efforts:
                        return (
                            False,
                            f"Traex 模型 {base_model}（profile={target_profile}）不支持设置 effort（收到 effort={effective_effort}）",
                        )
                    traex_efforts = prof_obj.reasoning_efforts
                elif (
                    cap.authoritative_models
                    and base_model in cap.models
                    and not cap.efforts_by_model.get(base_model)
                ):
                    return (
                        False,
                        f"Traex 模型 {base_model} 不支持设置 effort（收到 effort={effective_effort}）",
                    )
                if traex_efforts is None:
                    traex_efforts = cap.efforts_by_model.get(base_model)
            allowed = set(traex_efforts) if traex_efforts else set(CODEX_REASONING_EFFORTS)
            if effective_effort not in allowed:
                return (
                    False,
                    f"工具 traex 不支持的 effort: {effective_effort}"
                    f"（可选: {', '.join(sorted(allowed))}）",
                )
        elif tool == "dsh":
            model_efforts = cap.efforts_by_model.get(base_model or "") if base_model else None
            if model_efforts:
                allowed = set(model_efforts) | {DSH_DEFAULT_REASONING}
            elif (
                cap.authoritative_models
                and base_model
                and not is_default_base
                and base_model in cap.models
            ):
                allowed = {DSH_DEFAULT_REASONING}
            else:
                allowed = set(CODEX_REASONING_EFFORTS) | {"off", DSH_DEFAULT_REASONING}
            if effective_effort not in allowed:
                return (
                    False,
                    f"工具 dsh 不支持的 effort: {effective_effort}"
                    f"（可选: {', '.join(sorted(allowed))}）",
                )

    return True, None


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
            for name in ("traex", "claude", "codex", "grok", "dsh", "agy", "aiden", "gemini", "coco"):
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
        from ..agent_session.backend_resolver import cli_command_for_agent, is_cli_backend
        from ..mode.backends import get_backend_spec
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

    for name, provider in providers.items():
        if require_available:
            available = False

            # User-facing selection cards need a fast answer. A binary on PATH
            # is enough to offer the tool; the actual ACP startup path still
            # performs its normal capability checks and fallbacks later.
            if is_cli_backend(name):
                cli_cmd = cli_command_for_agent(name)
                available = shutil.which(cli_cmd) is not None
            else:
                cfg_bin = getattr(getattr(provider, "_config", None), "binary_name", None)
                spec = get_backend_spec(name)
                binary = (
                    cfg_bin
                    if isinstance(cfg_bin, str) and cfg_bin.strip()
                    else None
                ) or (spec.cli_name if spec is not None else None) or name
                if shutil.which(binary) is not None or (
                    binary != name and shutil.which(name) is not None
                ):
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
                    get_fallback = getattr(provider, "get_fallback_command", None)
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
