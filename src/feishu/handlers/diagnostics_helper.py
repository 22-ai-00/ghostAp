from typing import TYPE_CHECKING

from ...card.builders.diagnostics import DiagnosticsBuilder
from ...card.models import EngineStatusEntry
from ...card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from ..handler_context import HandlerContext

class DiagnosticsHelper:
    @staticmethod
    def get_all_engine_statuses(ctx: "HandlerContext", chat_id: str, include_done: bool = False) -> list[EngineStatusEntry]:
        """Collect and aggregate status entries from all engines (Deep, Spec)."""
        entries: list[EngineStatusEntry] = []

        deep_label = UI_TEXT["diag_engine_deep"]
        spec_label = UI_TEXT["diag_engine_spec"]

        # Deep engines
        for engine in ctx.deep_engine_manager.list_engines(chat_id):
            if not engine.project:
                continue
            p = engine.project
            status_val = p.status.value
            if not include_done and status_val in ("completed", "failed"):
                continue
            tid = p.task_id or ""
            info = DiagnosticsBuilder.format_engine_status_info(deep_label, p)
            entries.append(EngineStatusEntry(
                mode=deep_label,
                task_id=tid,
                name=p.name,
                status=status_val,
                info=info,
                started_at=p.started_at
            ))

        # Spec engines
        for engine in ctx.spec_engine_manager.list_engines(chat_id):
            if not engine.project:
                continue
            p = engine.project
            status_val = p.status.value
            if not include_done and status_val in ("completed", "aborted"):
                continue
            tid = p.task_id or ""
            info = DiagnosticsBuilder.format_engine_status_info(spec_label, p)
            entries.append(EngineStatusEntry(
                mode=spec_label,
                task_id=tid,
                name=p.name,
                status=status_val,
                info=info,
                started_at=p.started_at
            ))

        # Sort: running first, then by start time descending
        def _sort_key(e: EngineStatusEntry):
            status_val = e.status
            running = 0 if status_val in ("executing", "running", "planning", "analyzing") else 1
            return (running, -(e.started_at or 0))

        entries.sort(key=_sort_key)
        return entries
