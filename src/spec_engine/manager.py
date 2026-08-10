import json
import logging
import os
from typing import Optional

from ..engine_base import BaseEngineManager
from ..utils.engine_identity import resolve_engine_identity
from .engine import SpecEngine
from .models import SpecProject
from .storage import state_path_candidates

logger = logging.getLogger(__name__)


class SpecEngineManager(BaseEngineManager["SpecEngine"]):
    """Manages SpecEngine instances per chat.

    Uses a secondary index (_chat_keys) to avoid O(n) full-table scans.
    """

    def _create_engine(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str,
        engine_name: str,
        model_name: Optional[str],
    ) -> "SpecEngine":
        return SpecEngine(
            chat_id=chat_id,
            root_path=root_path,
            agent_type=agent_type,
            engine_name=engine_name,
            model_name=model_name,
        )

    def _resolve_engine_identity(
        self,
        *,
        engine_name: str = "Coco",
        agent_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> tuple[str, str, Optional[str]]:
        normalized_agent = str(agent_type or "").strip().lower()
        normalized_name = str(engine_name or "").strip() or "Coco"
        from ..mode import InteractionMode
        mode_map = {
            "coco": InteractionMode.COCO,
            "claude": InteractionMode.CLAUDE,
            "aiden": InteractionMode.AIDEN,
            "codex": InteractionMode.CODEX,
            "gemini": InteractionMode.GEMINI,
            "traex": InteractionMode.TRAEX,
        }
        backend = normalized_agent or next(
            (name for name in mode_map if normalized_name.lower().startswith(name)),
            "coco",
        )
        if backend not in mode_map:
            return SpecEngine._infer_engine_name(backend), backend, model_name
        if backend == "claude" and normalized_agent:
            return "Claude", "claude", None
        identity = resolve_engine_identity(mode=mode_map[backend])
        resolved_model = identity.model_name if backend == "claude" else (model_name or identity.model_name)
        return identity.engine_name, identity.agent_type, resolved_model

    def get_or_create(
        self,
        chat_id: str,
        root_path: str,
        engine_name: str = "Coco",
        *,
        agent_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "SpecEngine":
        key = f"{chat_id}:{root_path}"
        resolved_engine_name, resolved_agent_type, resolved_model_name = self._resolve_engine_identity(
            engine_name=engine_name,
            agent_type=agent_type,
            model_name=model_name,
        )

        with self._lock:
            existing = self._engines.get(key)
            should_replace = False
            if existing:
                should_replace = (
                    not existing.is_running
                    and (
                        existing.engine_name.lower() != resolved_engine_name.lower()
                        or existing._agent_type != resolved_agent_type
                        or existing._model_name != resolved_model_name
                    )
                )

            if existing is None or should_replace:
                if existing is not None:
                    existing.cleanup()
                self._engines[key] = self._create_engine(
                    chat_id, root_path, resolved_agent_type,
                    resolved_engine_name, resolved_model_name,
                )
                self._add_index(chat_id, key)

            return self._engines[key]

    def load_or_create_from_disk(self, chat_id: str, root_path: str, engine_name: str = "Coco") -> "SpecEngine":
        """Hydrate the one canonical run-state for automatic recovery/reporting."""
        seed_engine = self.get_or_create(chat_id, root_path, engine_name=engine_name)
        if not seed_engine.settings.spec_allow_resume_from_disk:
            return seed_engine

        return self._load_or_create_from_state_paths(
            chat_id,
            root_path,
            state_path_candidates(root_path, seed_engine.settings),
            engine_name=engine_name,
        )

    def _load_or_create_from_state_paths(
        self,
        chat_id: str,
        root_path: str,
        state_paths: list[str],
        *,
        engine_name: str,
    ) -> "SpecEngine":
        self.get_or_create(chat_id, root_path, engine_name=engine_name)
        persisted_project = None
        persisted_runtime = None
        persisted_saved_at = None
        persisted_compact = None
        loaded_state_path = ""

        for state_path in state_paths:
            if not state_path or not os.path.exists(state_path):
                continue
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                proj = data.get("project")
                if isinstance(proj, dict):
                    persisted_project = proj
                    persisted_runtime = data.get("runtime_context") if isinstance(data.get("runtime_context"), dict) else None
                    persisted_saved_at = data.get("saved_at")
                    persisted_compact = proj.get("_compact")
                    loaded_state_path = state_path
                    break
            except Exception:
                persisted_project = None
                persisted_runtime = None

        runtime = dict(persisted_runtime or {})
        engine = self.get_or_create(
            chat_id,
            root_path,
            engine_name=runtime.get("engine_name") or engine_name,
            agent_type=runtime.get("agent_type"),
            model_name=runtime.get("model_name") or runtime.get("current_model"),
        )

        if persisted_project is not None and engine.project is None:
            try:
                engine._project = SpecProject.from_dict(persisted_project)
                engine._restore_runtime_context(runtime)
                engine._resume_meta = {
                    "state_path": loaded_state_path,
                    "saved_at": persisted_saved_at,
                    "compact": persisted_compact,
                }
            except Exception:
                logger.debug("failed to attach persistence metadata", exc_info=True)
        return engine

    def _build_snapshot(self, engine: "SpecEngine"):
        """Build EngineSnapshot with Spec-specific fields."""
        from src.card.engine_snapshot import EngineSnapshot
        project = engine.project
        return EngineSnapshot(
            engine_name=engine.engine_name,
            root_path=engine.root_path,
            satisfied_count=project.satisfied_count if project else 0,
            total_criteria=project.total_criteria if project else 0,
            duration_seconds=project.duration() if project else None,
            status=project.status.value if project else "",
            is_running=engine.is_running,
            cycle_count=len(project.cycles) if project else 0,
            cycle_count_total=project.cycle_count_total if project else 0,
            ext={"project": project},
        )
