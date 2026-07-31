"""WorkflowEngineManager — manages per-chat WorkflowEngine instances."""

from __future__ import annotations

import logging
from typing import Optional

from ..engine_base import BaseEngineManager
from .engine import WorkflowEngine

logger = logging.getLogger(__name__)


class WorkflowEngineManager(BaseEngineManager["WorkflowEngine"]):
    """Manages WorkflowEngine instances keyed by chat_id:root_path.

    Provides get_or_create, lookup, and cleanup for workflow engines.
    """

    def _create_engine(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str,
        engine_name: str,
        model_name: Optional[str],
    ) -> WorkflowEngine:
        """Factory method — create a new WorkflowEngine instance."""
        return WorkflowEngine(
            chat_id=chat_id,
            root_path=root_path,
            agent_type=agent_type,
            engine_name=engine_name,
            model_name=model_name,
        )

    def _build_snapshot(self, engine: WorkflowEngine) -> dict:
        """Build a status snapshot for rendering/diagnostics."""
        project = engine.workflow_project
        return {
            "chat_id": engine.chat_id,
            "root_path": engine.root_path,
            "engine_name": engine.engine_name,
            "run_state": engine.run_state.value,
            "workflow_id": project.workflow_id if project else None,
            "status": project.status.value if project else "idle",
            "name": project.name if project else None,
            "metrics": project.metrics.model_dump() if project else None,
        }

    def remove(self, chat_id: str, root_path: str) -> None:
        """Remove and cleanup a workflow engine instance."""
        key = f"{chat_id}:{root_path}"
        with self._lock:
            engine = self._engines.get(key)
            if not engine:
                return
            try:
                quiesced = engine.cleanup()
            except Exception as e:
                logger.debug("Cleanup failed for %s: %s", key, repr(e))
                quiesced = False
            if quiesced is not False:
                self._engines.pop(key, None)
                self._remove_index(chat_id, key)
