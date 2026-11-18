"""Manager for per-chat/project DeepEngine instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..engine_base import BaseEngineManager

if TYPE_CHECKING:
    from .engine import DeepEngine


class DeepEngineManager(BaseEngineManager["DeepEngine"]):
    """Manages DeepEngine instances per chat+project.

    Thread-safe: all dict mutations are protected by _lock.
    Uses a secondary index (_chat_keys) to avoid O(n) full-table scans.
    """

    def _create_engine(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str,
        engine_name: str,
        model_name: Optional[str],
    ) -> "DeepEngine":
        from .engine import DeepEngine

        return DeepEngine(
            chat_id=chat_id,
            root_path=root_path,
            agent_type=agent_type,
            engine_name=engine_name,
            model_name=model_name,
        )

    def remove(self, chat_id: str, root_path: str):
        key = f"{chat_id}:{root_path}"
        with self._lock:
            if key in self._engines:
                self._engines[key].cleanup()
                del self._engines[key]
                self._remove_index(chat_id, key)

    def find_by_deep_project_id(self, chat_id: str, deep_project_id: str) -> Optional["DeepEngine"]:
        if not deep_project_id:
            return None
        for engine in self._iter_chat_engines(chat_id):
            try:
                if engine.project and engine.project.project_id == deep_project_id:
                    return engine
            except Exception:
                continue
        return None

    def _build_snapshot(self, engine: "DeepEngine"):
        """Build EngineSnapshot with Deep-specific fields."""
        from src.card.engine_snapshot import EngineSnapshot

        project = engine.project
        progress = engine.progress
        return EngineSnapshot(
            engine_name=engine.engine_name,
            root_path=engine.root_path,
            project_id=project.project_id if project else "",
            tool_calls_count=len(progress.tool_calls) if progress else 0,
            completed_steps=progress.completed_steps if progress else 0,
            total_steps=progress.total_steps if progress else 0,
            duration_seconds=project.duration() if project else None,
            status=project.status.value if project else "",
            is_running=engine.is_running,
            ext={"project": project, "progress": progress},
        )
