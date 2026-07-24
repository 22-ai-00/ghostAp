import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from ..tasking import TaskEvent, TaskPriority, TaskSpec, TaskStatus

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _PendingExit:
    chat_id: str
    project_id: Optional[str]
    message_id: str
    requested_at: float

class ControlPlane:
    """Handles control-plane logic like deferred /exit and system command gating."""

    def __init__(self, scheduler, project_manager, exit_handler_fn):
        self._scheduler = scheduler
        self._project_manager = project_manager
        self._exit_handler_fn = exit_handler_fn

        self._pending_exit_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._pending_exits: dict[str, _PendingExit] = {}  # key -> pending exit

        self._event_q: Deque[str] = deque()
        self._event_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._wakeup = threading.Event()
        self._stop_event = threading.Event()

        self._system_cmd_gate_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._system_cmd_inflight_by_chat: dict[str, int] = {}
        self._system_cmd_runs_by_chat: dict[str, set[str]] = {}

        self._thread = threading.Thread(
            target=self._loop,
            name="control_plane",
            daemon=True,
        )
        self._thread.start()

    def is_system_cmd_inflight(self, chat_id: str) -> bool:
        with self._system_cmd_gate_lock:
            return chat_id in self._system_cmd_inflight_by_chat

    def on_scheduler_event(self, ev: TaskEvent) -> None:
        """TaskScheduler listener (MUST be non-blocking)."""
        try:
            # 1) System command gate state
            if ev.task_type in {"system_help", "system_exit"}:
                terminal = {
                    TaskStatus.SUCCEEDED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELED,
                }
                with self._system_cmd_gate_lock:
                    runs = self._system_cmd_runs_by_chat.setdefault(ev.chat_id, set())
                    if ev.status == TaskStatus.RUNNING:
                        runs.add(ev.run_id)
                    elif ev.status in terminal:
                        runs.discard(ev.run_id)
                    if runs:
                        self._system_cmd_inflight_by_chat[ev.chat_id] = len(runs)
                    else:
                        self._system_cmd_runs_by_chat.pop(ev.chat_id, None)
                        self._system_cmd_inflight_by_chat.pop(ev.chat_id, None)

            # 2) Deferred exit processing wakeup
            if ev.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}:
                return
            key = self._pending_exit_key(ev.chat_id, ev.project_id)
            with self._event_lock:
                self._event_q.append(key)
            self._wakeup.set()
        except Exception:
            logger.debug("Unexpected error in on_scheduler_event", exc_info=True)
            return

    @staticmethod
    def _pending_exit_key(chat_id: str, project_id: Optional[str]) -> str:
        return f"{chat_id}:{project_id or 'DEFAULT'}"

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._wakeup.wait(timeout=0.2)
            if self._stop_event.is_set():
                return

            keys: set[str] = set()
            with self._event_lock:
                while self._event_q:
                    keys.add(self._event_q.popleft())
                self._wakeup.clear()

            for key in keys:
                try:
                    self._maybe_finalize_deferred_exit(key)
                except Exception:
                    continue

    def _maybe_finalize_deferred_exit(self, key: str) -> None:
        with self._pending_exit_lock:
            pending = self._pending_exits.get(key)
        if not pending:
            return

        # Check if any non-system task is still running under the same scope.
        project_id = pending.project_id
        tasks = self._scheduler.list_tasks(chat_id=pending.chat_id, project_id=project_id, include_done=False, limit=200)
        has_running_non_system = any(
            (st.status == TaskStatus.RUNNING) and (not bool(getattr(st.spec, "is_system_command", False)))
            for st in tasks
        )
        if has_running_non_system:
            return

        with self._pending_exit_lock:
            pending = self._pending_exits.pop(key, None)
        if not pending:
            return

        def _do_exit(_ctx):
            proj = None
            try:
                if pending.project_id:
                    proj = self._project_manager.get_project_for_chat(pending.project_id, pending.chat_id)
                if proj is None:
                    proj = self._project_manager.get_active_project(pending.chat_id)
            except Exception:
                proj = None
            self._exit_handler_fn(pending.message_id, pending.chat_id, project=proj)
            return True

        spec = TaskSpec(
            chat_id=pending.chat_id,
            name="deferred_exit",
            task_type="system_exit",
            message_id=pending.message_id,
            project_id=pending.project_id,
            origin_message_id=pending.message_id,
            priority=TaskPriority.HIGH,
            is_system_command=True,
        )
        self._scheduler.submit(spec, _do_exit)

    def request_deferred_exit(self, *, message_id: str, chat_id: str, project_id: Optional[str]) -> None:
        key = self._pending_exit_key(chat_id, project_id)
        with self._pending_exit_lock:
            self._pending_exits[key] = _PendingExit(
                chat_id=chat_id,
                project_id=project_id,
                message_id=message_id,
                requested_at=time.time(),
            )

    def should_defer_exit(self, *, chat_id: str, project_id: Optional[str]) -> bool:
        tasks = self._scheduler.list_tasks(chat_id=chat_id, project_id=project_id, include_done=False, limit=200)
        return any(
            (st.status == TaskStatus.RUNNING) and (not bool(getattr(st.spec, "is_system_command", False)))
            for st in tasks
        )

    def stop(self):
        self._stop_event.set()
        self._wakeup.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
