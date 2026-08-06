import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..utils.errors import get_error_detail

logger = logging.getLogger(__name__)

SPEC_TASKS_DIR = os.path.expanduser("~/.cache/ghostAp/spec_tasks")
SPEC_TASKS_DIR_FALLBACK = os.path.join(tempfile.gettempdir(), "ghostap_spec_tasks")


def _iter_task_dirs() -> list[str]:
    dirs = []
    for d in (SPEC_TASKS_DIR, SPEC_TASKS_DIR_FALLBACK):
        if d and d not in dirs:
            dirs.append(d)
    return dirs


@dataclass
class SpecTaskState:
    task_id: str
    created_at: float
    requirement: str
    project_path: str
    chat_id: str
    agent_type: str
    current_cycle: int
    current_phase: str
    last_error: str
    retry_count: int
    # 任务状态（用于恢复列表/详情展示）
    status: str = "失败"
    # 失败原因（面向用户/验收的稳定字段），例如："Phase build 失败: Internal error"
    failure_reason: str = ""
    models_tried: list[str] = field(default_factory=list)
    project_snapshot: Optional[dict] = None
    runtime_context: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "requirement": self.requirement,
            "project_path": self.project_path,
            "chat_id": self.chat_id,
            "agent_type": self.agent_type,
            "current_cycle": self.current_cycle,
            "current_phase": self.current_phase,
            "last_error": self.last_error,
            "retry_count": self.retry_count,
            "models_tried": self.models_tried,
            "project_snapshot": self.project_snapshot,
            "runtime_context": self.runtime_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecTaskState":
        task_id = data["task_id"]
        created_at = data["created_at"]
        requirement = data.get("requirement", "")
        project_path = data.get("project_path", "")
        chat_id = data.get("chat_id", "")
        agent_type = data.get("agent_type", "")
        current_cycle = int(data.get("current_cycle") or 0)
        current_phase = data.get("current_phase", "")
        last_error = data.get("last_error", "")
        retry_count = int(data.get("retry_count") or 0)
        models_tried = data.get("models_tried") or []
        project_snapshot = data.get("project_snapshot")
        runtime_context = data.get("runtime_context") if isinstance(data.get("runtime_context"), dict) else None

        status = (data.get("status") or "").strip() or "失败"
        failure_reason = (data.get("failure_reason") or "").strip()
        if not failure_reason:
            phase = str(current_phase or "").strip() or ""
            err = str(last_error or "").strip() or ""
            if phase and err:
                # Backward compat: older snapshots only have current_phase/last_error.
                failure_reason = f"Phase {phase} 失败: {err}"

        return cls(
            task_id=task_id,
            created_at=created_at,
            requirement=requirement,
            project_path=project_path,
            chat_id=chat_id,
            agent_type=agent_type,
            current_cycle=current_cycle,
            current_phase=current_phase,
            last_error=last_error,
            retry_count=retry_count,
            status=status,
            failure_reason=failure_reason,
            models_tried=models_tried,
            project_snapshot=project_snapshot,
            runtime_context=runtime_context,
        )

    def resolved_runtime_context(self) -> dict:
        ctx = dict(self.runtime_context or {})
        agent_type = str(ctx.get("agent_type") or self.agent_type or "coco").strip().lower() or "coco"
        engine_name = str(ctx.get("engine_name") or "").strip()
        if not engine_name:
            if agent_type == "claude":
                engine_name = "Claude"
            else:
                engine_name = "Coco"

        models_tried = [str(model).strip() for model in (ctx.get("models_tried") or self.models_tried or []) if str(model).strip()]
        current_model = str(ctx.get("current_model") or "").strip()
        if not current_model and models_tried:
            current_model = models_tried[-1]

        model_name = str(ctx.get("model_name") or "").strip()
        if not model_name:
            model_name = current_model

        return {
            "agent_type": agent_type,
            "engine_name": engine_name,
            "model_name": model_name or None,
            "current_model": current_model or None,
            "models_tried": models_tried,
        }

    def resolved_engine_name(self) -> str:
        return str(self.resolved_runtime_context().get("engine_name") or "Coco")

    def resolved_model_name(self) -> Optional[str]:
        return self.resolved_runtime_context().get("model_name")

    def normalized_agent_type(self) -> str:
        return str(self.resolved_runtime_context().get("agent_type") or "coco")


def generate_task_id() -> str:
    return str(uuid.uuid4())[:8]


def save_task_state(state: SpecTaskState) -> str:
    last_err: Optional[Exception] = None
    for root in _iter_task_dirs():
        filepath = os.path.join(root, f"{state.task_id}.json")
        tmp_path = filepath + ".tmp"
        try:
            os.makedirs(root, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
            logger.debug("保存任务状态: %s", filepath)
            return filepath
        except Exception as e:
            last_err = e
            logger.warning("保存任务状态失败 %s: %s", root, get_error_detail(e))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.debug("failed to delete temp file", exc_info=True)
    if last_err:
        logger.warning("保存任务状态失败: %s", get_error_detail(last_err))
    return ""


def load_task_state(task_id: str) -> Optional[SpecTaskState]:
    for root in _iter_task_dirs():
        filepath = os.path.join(root, f"{task_id}.json")
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SpecTaskState.from_dict(data)
        except Exception as e:
            logger.warning("加载任务状态失败 %s: %s", task_id, get_error_detail(e))
            continue
    return None


def delete_task_state(task_id: str) -> bool:
    deleted = False
    for root in _iter_task_dirs():
        filepath = os.path.join(root, f"{task_id}.json")
        if not os.path.exists(filepath):
            continue
        try:
            os.unlink(filepath)
            logger.debug("删除任务状态: %s", filepath)
            deleted = True
        except OSError as e:
            logger.warning("删除任务状态失败 %s: %s", task_id, get_error_detail(e))
    return deleted


def list_pending_tasks() -> list[SpecTaskState]:
    tasks = []
    seen: set[str] = set()
    for root in _iter_task_dirs():
        if not os.path.isdir(root):
            continue
        for filename in os.listdir(root):
            if not filename.endswith(".json"):
                continue
            task_id = filename[:-5]
            if task_id in seen:
                continue
            state = load_task_state(task_id)
            if state:
                tasks.append(state)
                seen.add(task_id)
    return tasks
