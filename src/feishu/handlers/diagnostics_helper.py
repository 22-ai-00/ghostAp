from __future__ import annotations

from typing import TYPE_CHECKING

from ...card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from ...card.models import EngineStatusEntry
    from ..handler_context import HandlerContext

class DiagnosticsHelper:
    @staticmethod
    def get_all_engine_statuses(ctx: "HandlerContext", chat_id: str, include_done: bool = False) -> list[EngineStatusEntry]:
        """Collect and aggregate status entries from all engines (Deep, Spec)."""
        from ...card.builders.diagnostics import DiagnosticsBuilder
        from ...card.models import EngineStatusEntry

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

        # Workflow engines
        wf_mgr = getattr(ctx, "workflow_engine_manager", None)
        if wf_mgr is not None and callable(getattr(wf_mgr, "list_engines", None)):
            wf_engines = wf_mgr.list_engines(chat_id)
            if isinstance(wf_engines, (list, tuple)):
                wf_label = UI_TEXT.get("diag_engine_workflow", "Workflow")
                for engine in wf_engines:
                    wp = getattr(engine, "workflow_project", None)
                    if not wp:
                        continue
                    status_obj = getattr(wp, "status", None)
                    status_val = getattr(status_obj, "value", str(status_obj or "idle"))
                    if not include_done and status_val in ("completed", "failed", "cancelled", "idle"):
                        continue
                    tid = getattr(wp, "workflow_id", "") or ""
                    phases = getattr(wp, "phases", ()) or ()
                    done_phases = sum(
                        1
                        for ph in phases
                        if (
                            getattr(ph, "finished_at", None) is not None
                            or getattr(
                                getattr(ph, "status", None),
                                "value",
                                getattr(ph, "status", ""),
                            )
                            == "completed"
                        )
                    )
                    parts = [status_val]
                    if phases:
                        parts.append(f"{done_phases}/{len(phases)} 阶段")
                    entries.append(
                        EngineStatusEntry(
                            mode=wf_label,
                            task_id=tid,
                            name=getattr(wp, "name", "") or "Workflow",
                            status=status_val,
                            info=" · ".join(parts),
                            started_at=getattr(wp, "started_at", None),
                        )
                    )

        # Sort: running first, then by start time descending
        def _sort_key(e: EngineStatusEntry):
            status_val = e.status
            running = 0 if status_val in ("executing", "running", "planning", "analyzing") else 1
            return (running, -(e.started_at or 0))

        entries.sort(key=_sort_key)
        return entries
