"""Persistence helpers for Spec Engine — state, artifacts, history."""

import json
import logging
import os
import re
import shutil
import time
from typing import Optional

from ..utils.errors import get_error_detail
from .models import SpecProject, SpecWorkItem, SpecWorkItemStatus
from .storage import (
    artifact_root_dir as _storage_artifact_root_dir,
)
from .storage import (
    get_state_path as _storage_get_state_path,
)
from .storage import (
    run_state_path as _storage_run_state_path,
)

logger = logging.getLogger(__name__)


def get_state_path(root_path: str, settings) -> str:
    return _storage_get_state_path(root_path, settings)


def persist_state_best_effort(project: Optional[SpecProject], save_fn, state_path: str) -> None:
    if not project:
        return
    try:
        save_fn(state_path)
    except Exception as e:
        logger.debug("[Spec] 保存状态失败: %s", get_error_detail(e))


def artifact_root_dir(root_path: str, settings, project: Optional[SpecProject]) -> str:
    pid = project.project_id if project else "unknown"
    return _storage_artifact_root_dir(root_path, settings, pid)


def history_log_path(root_path: str, settings, project: Optional[SpecProject]) -> str:
    root = artifact_root_dir(root_path, settings, project)
    os.makedirs(root, exist_ok=True)
    filename = settings.spec_history_log_filename
    return os.path.join(root, filename)


def append_history_event(
    root_path: str,
    settings,
    project: Optional[SpecProject],
    event_type: str,
    payload: dict,
) -> None:
    try:
        path = history_log_path(root_path, settings, project)
        record = {
            "ts": time.time(),
            "type": event_type,
            "project_id": project.project_id if project else "",
            "cycle": int(payload.get("cycle") or 0),
            "payload": payload,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("[Spec] Failed to append history event (type=%s)", event_type, exc_info=True)
        return


def persist_cycle_artifact(
    root_path: str,
    settings,
    project: Optional[SpecProject],
    cycle_num: int,
    name: str,
    content: str,
    ext: str = "txt",
) -> Optional[str]:
    if not content and name not in ("metrics", "discovery"):
        return None
    try:
        root = artifact_root_dir(root_path, settings, project)
        cycle_dir = os.path.join(root, f"cycle_{cycle_num:04d}")
        os.makedirs(cycle_dir, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
        path = os.path.join(cycle_dir, f"{safe_name}.{ext}")
        tmp = path + ".tmp"

        persist_max = settings.spec_phase_output_persist_max_chars
        to_write = content or ""
        if persist_max > 0 and len(to_write) > persist_max:
            to_write = to_write[:persist_max] + "\n...\n(已截断，超长输出未全部落盘)"

        with open(tmp, "w", encoding="utf-8") as f:
            f.write(to_write)
        os.replace(tmp, path)
        return path
    except Exception as e:
        logger.debug("[Spec] 落盘产物失败(%s): %s", name, get_error_detail(e))
        return None


def cleanup_old_cycle_artifacts(
    root_path: str,
    settings,
    project: Optional[SpecProject],
    current_cycle: int,
) -> None:
    try:
        retention = settings.spec_cycle_artifact_retention
        if retention <= 0:
            return
        root = artifact_root_dir(root_path, settings, project)
        cutoff = current_cycle - retention
        if cutoff <= 0:
            return
        old_dir = os.path.join(root, f"cycle_{cutoff:04d}")
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)

        if project and project.cycles:
            for c in project.cycles:
                if c.cycle_number == cutoff:
                    c.spec_path = None
                    c.plan_path = None
                    c.tasks_path = None
                    c.build_path = None
                    c.review_path = None
                    c.discovery_path = None
                    c.metrics_path = None
                    break
    except Exception:
        logger.debug("[Spec] Failed to cleanup old cycle artifacts (cycle=%s)", current_cycle, exc_info=True)
        return


def cleanup_generated_specs(project: Optional[SpecProject], settings) -> None:
    if not project:
        return
    try:
        retention = settings.spec_generated_specs_retention
        if retention <= 0:
            return

        keep_paths: set[str] = set()
        for w in project.work_items:
            if w.status in (SpecWorkItemStatus.PENDING, SpecWorkItemStatus.IN_PROGRESS):
                if w.spec_path:
                    keep_paths.add(w.spec_path)

        done = [w for w in project.work_items if w.status == SpecWorkItemStatus.DONE and w.spec_path]
        done.sort(key=lambda x: x.created_at)
        keep_done = done[-retention:]
        for w in keep_done:
            keep_paths.add(w.spec_path)

        for w in done[:-retention]:
            p = w.spec_path
            if not p or p in keep_paths:
                continue
            try:
                if os.path.exists(p):
                    os.remove(p)
                w.spec_deleted = True
            except Exception:
                continue
    except Exception:
        logger.debug("[Spec] Failed to cleanup generated specs", exc_info=True)
        return


def persist_generated_spec_file(
    root_path: str,
    settings,
    project: Optional[SpecProject],
    cycle_num: int,
    qid: str,
    spec_text: str,
) -> str:
    try:
        root = artifact_root_dir(root_path, settings, project)
        spec_dir = os.path.join(root, "generated_specs")
        os.makedirs(spec_dir, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", qid)[:80] or f"Q_{cycle_num}"
        path = os.path.join(spec_dir, f"cycle_{cycle_num:04d}_{safe_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(spec_text or "{}")
        os.replace(tmp, path)
        return path
    except Exception as e:
        logger.warning("[Spec] 生成 spec 文件落盘失败(cycle=%s, qid=%s): %s", cycle_num, qid, get_error_detail(e))
        return ""


def read_text_file_best_effort(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def truncate_output(text: str, settings) -> str:
    max_chars = settings.spec_cycle_output_max_chars
    if max_chars <= 0:
        return text or ""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...\n(已截断，完整内容见落盘产物)"


def project_to_compact_dict(
    project: SpecProject,
    settings,
    root_path: str,
) -> dict:
    if not project:
        return {}

    tail_cycles = int(settings.spec_state_cycles_tail or 50)
    tail_items = int(settings.spec_state_work_items_tail or 200)
    tail_metrics = int(settings.spec_state_metrics_tail or 200)

    cycle_count_total = max(int(getattr(project, "cycle_count_total", 0) or 0), len(project.cycles))
    work_items_total = max(int(getattr(project, "work_items_total", 0) or 0), len(project.work_items))

    pd = {
        "project_id": project.project_id,
        "name": project.name,
        "root_path": project.root_path,
        "requirement": project.requirement,
        "acceptance_criteria": list(project.acceptance_criteria),
        "criteria_tracker": project.criteria_tracker.to_dict(),
        "status": project.status.value,
        "created_at": project.created_at,
        "started_at": project.started_at,
        "completed_at": project.completed_at,
        "error": project.error,
        "auto_recovery_attempts": project.auto_recovery_attempts,
        "cycle_count_total": cycle_count_total,
        "work_items_total": work_items_total,
        "cycles": [c.to_dict() for c in (project.cycles[-tail_cycles:] if tail_cycles > 0 else [])],
        "metrics_history": [
            m.to_dict() for m in (project.metrics_history[-tail_metrics:] if tail_metrics > 0 else [])
        ],
    }

    pending = [w for w in project.work_items if w.status == SpecWorkItemStatus.PENDING]
    recent = project.work_items[-tail_items:] if tail_items > 0 else []
    merged: list[SpecWorkItem] = []
    seen: set[str] = set()
    for w in pending + list(recent):
        if not w or not w.item_id or w.item_id in seen:
            continue
        seen.add(w.item_id)
        merged.append(w)
    pd["work_items"] = [w.to_dict() for w in merged]
    pd["_compact"] = {
        "cycles_tail": tail_cycles,
        "work_items_tail": tail_items,
        "metrics_tail": tail_metrics,
        "cycles_truncated_before": max(0, cycle_count_total - len(pd.get("cycles") or [])),
    }
    pd["artifacts_root"] = artifact_root_dir(root_path, settings, project)
    pd["history_log_path"] = history_log_path(root_path, settings, project)
    return pd


def save_engine_state(
    project: SpecProject,
    settings,
    root_path: str,
    chat_id: str,
    build_runtime_context_fn,
    project_to_compact_dict_fn,
    filepath: Optional[str] = None,
    review_circuit: Optional[dict] = None,
) -> str:
    if not project:
        raise ValueError("没有项目状态可保存")
    default_path = not filepath
    if not filepath:
        filepath = get_state_path(root_path, settings)
    state = {
        "chat_id": chat_id,
        "root_path": root_path,
        "project": project_to_compact_dict_fn(),
        "runtime_context": build_runtime_context_fn(),
        "saved_at": time.time(),
    }
    if review_circuit:
        state["review_circuit"] = review_circuit
    if default_path:
        run_path = _storage_run_state_path(root_path, settings, project.project_id)
        if run_path != filepath:
            _write_json_atomic(run_path, state)
    _write_json_atomic(filepath, state)
    return filepath


def _write_json_atomic(filepath: str, data: dict) -> None:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, filepath)


def load_engine_state(filepath: str) -> tuple[Optional[SpecProject], dict]:
    """Load engine state.  Returns (project, review_circuit_dict).

    ``review_circuit_dict`` is empty when the snapshot predates circuit
    persistence (backward-compatible).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        proj = data.get("project")
        if not isinstance(proj, dict):
            return None, {}
        rc = data.get("review_circuit")
        return SpecProject.from_dict(proj), rc if isinstance(rc, dict) else {}
    except Exception:
        return None, {}
