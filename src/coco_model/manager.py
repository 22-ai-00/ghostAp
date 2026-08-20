import logging
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from acp.stdio import spawn_agent_process

from ..acp.transport import LateFrameTolerantMessageQueue
from ..utils.async_helpers import safe_wait_for
from ..utils.errors import get_error_detail
from .models import CocoModel, ModelListResult

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    CocoModel(name="gpt-5.2", description="GPT-5.2"),
    CocoModel(name="gpt-4.1", description="GPT-4.1"),
    CocoModel(name="claude-3-opus", description="Claude 3 Opus"),
    CocoModel(name="claude-3.5-sonnet", description="Claude 3.5 Sonnet"),
    CocoModel(name="claude-3.7-sonnet", description="Claude 3.7 Sonnet"),
    CocoModel(name="doubao-1.5-pro", description="Doubao 1.5 Pro"),
]

CACHE_TTL_SECONDS = 1800
# When the ACP probe fails and we degrade to the static DEFAULT_MODELS list, we
# must NOT pin that stale list for the full 30 minutes.  A five-minute retry
# window avoids repeatedly starting Coco while keeping explicit refresh useful.
FALLBACK_CACHE_TTL_SECONDS = 300


class CocoModelManager:
    def __init__(self):
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._cached_models: Optional[list[CocoModel]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = CACHE_TTL_SECONDS
        self._current_model: Optional[str] = None
        self._config_path = Path.home() / "Library" / "Application Support" / "coco" / "coco.yaml"
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._current_model = self._read_model_from_config()
            self._initialized = True

    def _read_model_from_config(self) -> Optional[str]:
        try:
            if not self._config_path.exists():
                logger.debug("coco config not found: %s", self._config_path)
                return None
            with open(self._config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config and isinstance(config.get("model"), dict):
                return config["model"].get("name")
        except Exception as e:
            logger.warning("Failed to read coco config: %s", get_error_detail(e))
        return None

    def _is_cache_valid(self) -> bool:
        return self._cached_models is not None and (time.time() - self._cache_time) < self._cache_ttl

    def _is_static_fallback(self, models: list[CocoModel]) -> bool:
        """True when ``models`` is just the hardcoded DEFAULT_MODELS list (probe failed)."""
        default_names = {m.name for m in DEFAULT_MODELS}
        return bool(models) and {m.name for m in models} == default_names

    def get_models(self) -> ModelListResult:
        self._ensure_initialized()
        with self._lock:
            if self._is_cache_valid():
                return ModelListResult(
                    models=list(self._cached_models),
                    cached=True,
                )
            try:
                models = self._load_models()
                self._cached_models = models
                self._cache_time = time.time()
                # Pin a real ACP-probed list for the full TTL; if the probe
                # failed and we only have the static defaults, expire fast so
                # the next caller (incl. the "刷新模型列表" button) retries.
                self._cache_ttl = (
                    FALLBACK_CACHE_TTL_SECONDS if self._is_static_fallback(models) else CACHE_TTL_SECONDS
                )
                return ModelListResult(models=list(models), cached=False)
            except Exception as e:
                logger.error("Failed to load coco models: %s", get_error_detail(e))
                return ModelListResult(
                    models=list(DEFAULT_MODELS),
                    cached=False,
                    error=get_error_detail(e),
                )

    def _load_models(self) -> list[CocoModel]:
        current = self._current_model or self._read_model_from_config()

        # ACP-first: use protocol-provided available models when possible.
        models_from_acp = self._load_models_via_acp(current_model=current)
        if models_from_acp:
            return models_from_acp

        models = []
        for m in DEFAULT_MODELS:
            models.append(
                CocoModel(
                    name=m.name,
                    description=m.description,
                    is_default=(m.name == current),
                )
            )
        return models

    def _load_models_via_acp(self, current_model: Optional[str]) -> list[CocoModel]:
        """Best-effort ACP model discovery for coco.

        For ACP-capable backends, model capabilities are exposed in new/load session
        responses (`SessionModelState.available_models`). We query once and map to
        CocoModel entries.
        """
        try:

            from src.acp.client import GhostAPClient

            async def _probe() -> list[CocoModel]:
                from src.utils.env import build_clean_env
                env = build_clean_env()

                client = GhostAPClient(on_event=lambda _ev: None)
                async with spawn_agent_process(
                    client,
                    "coco",
                    "acp",
                    "serve",
                    env=env,
                    cwd=str(Path.cwd()),
                    queue=LateFrameTolerantMessageQueue(),
                ) as (
                    conn,
                    _proc,
                ):
                    await conn.initialize(protocol_version=1)
                    resp = await conn.new_session(cwd=str(Path.cwd()))
                    models_state = getattr(resp, "models", None)
                    available = list(getattr(models_state, "available_models", []) or [])
                    current_id = str(
                        getattr(models_state, "current_model_id", "") or getattr(models_state, "currentModelId", "")
                    )

                    out: list[CocoModel] = []
                    for item in available:
                        model_id = str(
                            getattr(item, "model_id", "") or getattr(item, "modelId", "") or getattr(item, "name", "")
                        ).strip()
                        if not model_id:
                            continue
                        desc = str(getattr(item, "description", "") or getattr(item, "name", "") or model_id)
                        out.append(
                            CocoModel(
                                name=model_id,
                                description=desc,
                                is_default=(model_id == (current_model or current_id)),
                            )
                        )
                    return out

            try:
                from ..config import get_settings

                settings = get_settings()
                # Prefer the dedicated model-probe timeout (5s default) so the
                # ACP probe has a fair window to return real models before we
                # fall back to DEFAULT_MODELS.
                timeout_s = float(
                    getattr(settings, "acp_model_probe_timeout", None)
                    or getattr(settings, "acp_healthcheck_timeout", 2.0)
                    or 2.0
                )
            except Exception:
                timeout_s = 6.0
            from src.utils.async_helpers import run_async
            return run_async(
                safe_wait_for(
                    _probe(),
                    timeout=max(0.1, timeout_s),
                    action="Coco ACP 模型探测",
                )
            )
        except Exception as e:
            logger.debug("Failed to load coco models via ACP: %s", get_error_detail(e))
            return []

    def get_current_model(self) -> Optional[str]:
        self._ensure_initialized()
        with self._lock:
            return self._current_model

    def set_model(self, model_name: str) -> bool:
        self._ensure_initialized()
        normalized = (model_name or "").strip()
        if not normalized:
            logger.warning("Unknown model: %s", model_name)
            return False

        # ACP-first: validate against runtime-discovered model list.
        # Keep a DEFAULT_MODELS fallback for offline/failed discovery scenarios.
        result = self.get_models()
        known_names = {m.name for m in (result.models or []) if getattr(m, "name", "")}
        if not known_names:
            known_names = {m.name for m in DEFAULT_MODELS}

        if normalized not in known_names:
            logger.warning("Unknown model: %s", normalized)
            return False

        with self._lock:
            self._current_model = normalized
            self._cached_models = None
            self._cache_time = 0

        logger.info("Switched coco model to: %s", normalized)
        return True

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cached_models = None
            self._cache_time = 0
            self._cache_ttl = CACHE_TTL_SECONDS

    def kickoff_preheat(self) -> None:
        """Best-effort background warm-up of the ACP-probed model list.

        ``coco acp serve`` cold-start + initialize handshake is slow and highly
        variable (4-12s observed), so probing it lazily on the first ``/coco``
        click frequently times out and degrades to the static DEFAULT_MODELS.
        Kicking it off at process startup populates the 30min cache so the
        interactive path normally just reads a fresh list.
        """

        def _run() -> None:
            try:
                result = self.get_models()
                logger.info(
                    "coco model preheat: %d models (cached=%s, error=%s)",
                    len(result.models or []),
                    result.cached,
                    result.error,
                )
            except Exception as e:
                logger.debug("coco model preheat failed: %s", get_error_detail(e))

        threading.Thread(target=_run, name="coco-model-preheat", daemon=True).start()


_manager: Optional[CocoModelManager] = None
_manager_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def get_coco_model_manager() -> CocoModelManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = CocoModelManager()
    return _manager


def _reset_coco_model_manager_for_testing() -> None:
    """Reset the global CocoModelManager singleton. **Test-only.**"""
    global _manager
    with _manager_lock:
        _manager = None
