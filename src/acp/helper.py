"""Small ACP discovery helpers used by explicit configuration surfaces."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path

from acp.stdio import spawn_agent_process

from ..config import get_settings
from ..utils.async_helpers import safe_wait_for
from ..utils.text import get_acp_result_header_text
from .client import GhostAPClient
from .options import ACPModelOption, ACPToolOption
from .providers import get_providers

logger = logging.getLogger(__name__)

_TOOLS = ("coco", "claude", "aiden", "codex", "gemini", "traex")
_PROBE_TTL = 300.0
_CODEX_PROBE_TTL = 1800.0
_NEGATIVE_TTL = 45.0

_probe_cache: dict[tuple[str, str], tuple[float, list[ACPModelOption]]] = {}
_negative_cache: dict[tuple[str, str], float] = {}
_inflight: dict[tuple[str, str], threading.Event] = {}
_cache_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _key(tool_name: str, cwd: str | None) -> tuple[str, str]:
    return str(tool_name or "").strip().lower(), str(cwd or "")


def _copy(models: list[ACPModelOption]) -> list[ACPModelOption]:
    return [dataclasses.replace(model) for model in models]


def _mark_default(
    models: list[ACPModelOption], current_model: str | None
) -> list[ACPModelOption]:
    copied = _copy(models)
    selected = str(current_model or "").strip()
    if not selected or not any(model.name == selected for model in copied):
        return copied
    return [
        dataclasses.replace(model, is_default=model.name == selected)
        for model in copied
    ]


def _cached(
    key: tuple[str, str], tool_name: str
) -> tuple[list[ACPModelOption] | None, bool]:
    now = time.time()
    ttl = _CODEX_PROBE_TTL if tool_name == "codex" else _PROBE_TTL
    with _cache_lock:
        entry = _probe_cache.get(key)
        if entry and now - entry[0] <= ttl:
            return _copy(entry[1]), False
        if entry:
            _probe_cache.pop(key, None)

        failed_at = _negative_cache.get(key)
        if failed_at and now - failed_at <= _NEGATIVE_TTL:
            return None, True
        if failed_at:
            _negative_cache.pop(key, None)
    return None, False


def _store(key: tuple[str, str], models: list[ACPModelOption]) -> None:
    with _cache_lock:
        if models:
            _probe_cache[key] = (time.time(), _copy(models))
            _negative_cache.pop(key, None)
        else:
            _negative_cache[key] = time.time()


def invalidate_acp_model_cache(
    tool_name: str, cwd: str | None = None
) -> None:
    key = _key(tool_name, cwd)
    with _cache_lock:
        _probe_cache.pop(key, None)
        _negative_cache.pop(key, None)


def list_acp_tools() -> list[ACPToolOption]:
    """Return the available members of the six supported ACP backends."""
    providers = get_providers()
    descriptions = get_acp_result_header_text()
    tools: list[ACPToolOption] = []
    for name in _TOOLS:
        provider = providers.get(name)
        if provider is None:
            continue
        try:
            available = bool(provider.check_availability())
        except Exception:
            available = False
            logger.debug("[ACP] availability check failed for %s", name)
        if available:
            tools.append(
                ACPToolOption(
                    name=name,
                    description=descriptions.get(f"tool_desc_{name}") or name,
                    is_default=name == "coco",
                )
            )
    return tools


def _probe_timeout(explicit: float | None) -> float:
    if explicit is not None:
        return max(0.1, float(explicit))
    try:
        return max(0.1, float(get_settings().acp_model_probe_timeout))
    except Exception:
        return 15.0


def _fallback(tool_name: str, current_model: str | None) -> list[ACPModelOption]:
    selected = str(current_model or "").strip()
    if not selected or tool_name == "codex":
        return []
    return [ACPModelOption(name=selected, description=selected, is_default=True)]


def _coco_models(current_model: str | None) -> list[ACPModelOption]:
    try:
        from ..coco_model import get_coco_model_manager

        manager = get_coco_model_manager()
        selected = str(current_model or manager.get_current_model() or "").strip()
        models: list[ACPModelOption] = []
        seen: set[str] = set()
        for raw in manager.get_models().models or []:
            name = str(getattr(raw, "name", "") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            models.append(
                ACPModelOption(
                    name=name,
                    description=str(getattr(raw, "description", "") or name),
                    is_default=(name == selected)
                    if selected
                    else bool(getattr(raw, "is_default", False)),
                )
            )
        return models
    except Exception:
        logger.debug("[ACP] coco model lookup failed")
        return []


def _config_models(response: object, selected: str) -> list[ACPModelOption]:
    """Read the standard model select option from a new-session response."""
    for wrapped in getattr(response, "config_options", None) or ():
        root = getattr(wrapped, "root", wrapped)
        if not (
            str(getattr(root, "id", "") or "") == "model"
            or str(getattr(root, "category", "") or "") == "model"
        ):
            continue
        default = selected or str(getattr(root, "current_value", "") or "").strip()
        options: list[object] = []
        for option in getattr(root, "options", None) or ():
            nested = getattr(option, "options", None)
            options.extend(list(nested) if nested is not None else [option])
        return _options(options, default, value_field="value")
    return []


def _options(
    values: list[object], selected: str, *, value_field: str = "model_id"
) -> list[ACPModelOption]:
    models: list[ACPModelOption] = []
    seen: set[str] = set()
    for value in values:
        name = str(getattr(value, value_field, "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        models.append(
            ACPModelOption(
                name=name,
                description=str(
                    getattr(value, "description", "")
                    or getattr(value, "name", "")
                    or name
                ).strip(),
                is_default=name == selected,
            )
        )
    return models


def _response_models(
    response: object, current_model: str | None
) -> list[ACPModelOption]:
    state = getattr(response, "models", None)
    selected = str(
        current_model or getattr(state, "current_model_id", "") or ""
    ).strip()
    models = _options(list(getattr(state, "available_models", None) or ()), selected)
    return models or _config_models(response, selected)


async def _probe_acp_models(
    tool_name: str, cwd: str | None, current_model: str | None = None
) -> list[ACPModelOption]:
    """Start one provider-neutral ACP session and read its declared models."""
    provider = get_providers().get(tool_name)
    if provider is None:
        return []
    command, args = provider.get_serve_command(None)

    from ..utils.env import build_clean_env

    client = GhostAPClient(on_event=lambda _event: None, auto_approve=False)
    async with spawn_agent_process(
        client,
        command,
        *args,
        env=build_clean_env(),
        cwd=cwd or str(Path.cwd()),
    ) as (connection, _process):
        await connection.initialize(protocol_version=1)
        response = await connection.new_session(cwd=cwd or str(Path.cwd()))
        return _response_models(response, current_model)


def _probe_blocking(
    tool_name: str,
    cwd: str | None,
    current_model: str | None,
    timeout: float,
) -> list[ACPModelOption]:
    from ..utils.async_helpers import run_async
    from ..utils.errors import get_error_detail

    try:
        return run_async(
            safe_wait_for(
                _probe_acp_models(tool_name, cwd, current_model),
                timeout=timeout,
                action=f"ACP {tool_name} 模型探测",
            )
        ) or []
    except Exception as exc:
        logger.warning(
            "[ACP] model probe failed tool=%s err=%s",
            tool_name,
            get_error_detail(exc),
        )
        return []


def fetch_acp_models(
    tool_name: str,
    cwd: str | None,
    current_model: str | None = None,
    probe_timeout: float | None = None,
) -> list[ACPModelOption]:
    """Read backend-declared models with bounded, single-flight caching."""
    tool = str(tool_name or "").strip().lower()
    if tool == "coco":
        models = _coco_models(current_model)
        if models:
            return models

    key = _key(tool, cwd)
    models, failed = _cached(key, tool)
    if models is not None:
        return _mark_default(models, current_model)
    if failed:
        return _fallback(tool, current_model)

    leader = False
    with _cache_lock:
        event = _inflight.get(key)
        if event is None:
            event = threading.Event()
            _inflight[key] = event
            leader = True

    timeout = _probe_timeout(probe_timeout)
    if not leader:
        event.wait(timeout=timeout + 5.0)
        models, _failed = _cached(key, tool)
        return (
            _mark_default(models, current_model)
            if models is not None
            else _fallback(tool, current_model)
        )

    try:
        models = _probe_blocking(tool, cwd, current_model, timeout)
        _store(key, models)
    finally:
        with _cache_lock:
            _inflight.pop(key, None)
        event.set()

    return _mark_default(models, current_model) if models else _fallback(tool, current_model)


def kickoff_acp_model_preheat(
    tool_names: list[str], cwd: str
) -> threading.Thread | None:
    """Populate model caches in one best-effort background thread."""
    tools = list(
        dict.fromkeys(
            name
            for raw in tool_names
            if (name := str(raw or "").strip().lower()) in _TOOLS
        )
    )
    if not tools:
        return None

    def preheat() -> None:
        for tool in tools:
            started = time.monotonic()
            try:
                models = fetch_acp_models(tool, cwd)
                logger.info(
                    "[ACP] model preheat tool=%s count=%d duration_ms=%.1f",
                    tool,
                    len(models),
                    (time.monotonic() - started) * 1000,
                )
            except Exception as exc:
                from ..utils.errors import get_error_detail

                logger.info(
                    "[ACP] model preheat tool=%s failed=%s duration_ms=%.1f",
                    tool,
                    get_error_detail(exc),
                    (time.monotonic() - started) * 1000,
                )

    thread = threading.Thread(target=preheat, name="acp-model-preheat", daemon=True)
    thread.start()
    return thread


class SessionKeyCodec:
    """Encode the internal chat/project/thread routing key."""

    DEFAULT_PROJECT_PLACEHOLDER = "_default_"

    @classmethod
    def encode(
        cls,
        chat_id: str,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        project = project_id or cls.DEFAULT_PROJECT_PLACEHOLDER
        base = f"{chat_id}:{project}"
        return f"{base}:t:{thread_id}" if thread_id else base

    @classmethod
    def decode(cls, key: str) -> tuple[str, str | None, str | None]:
        try:
            value = str(key or "")
        except Exception:
            return "", None, None
        if not value:
            return "", None, None
        parts = value.split(":")
        chat_id = parts[0]
        project = parts[1] if len(parts) > 1 else ""
        project_id = (
            project if project and project != cls.DEFAULT_PROJECT_PLACEHOLDER else None
        )
        thread_id = parts[3] if len(parts) >= 4 and parts[2] == "t" else None
        return chat_id, project_id, thread_id
