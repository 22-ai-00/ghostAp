import contextvars
import logging
import math
import queue
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from enum import Enum, IntEnum
from typing import Any, Callable, ContextManager, Deque, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException, CircuitState
from ..utils.errors import get_error_detail
from ..utils.rate_limit import RateLimiter, RateLimitExceededException

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskPriority(IntEnum):
    HIGH = 0
    NORMAL = 10
    LOW = 20


SYSTEM_QUEUE_SUFFIX = ":SYSTEM"
DEFAULT_QUEUE_SUFFIX = ":DEFAULT"
_CURRENT_TASK_RUN_ID: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "ghostap_current_task_run_id",
        default=None,
    )
)


def get_current_task_run_id() -> str | None:
    """Return the scheduler run owning the current callback, if any."""

    return _CURRENT_TASK_RUN_ID.get()


class TaskSpec(BaseModel):
    """Metadata that influences routing and scheduling."""
    model_config = ConfigDict(frozen=True)

    chat_id: str
    name: str
    task_type: str = "generic"
    queue_key: Optional[str] = None
    project_id: Optional[str] = None
    message_id: Optional[str] = None
    origin_message_id: Optional[str] = None
    request_id: Optional[str] = None
    task_id: Optional[str] = None  # Human-readable ID e.g. "myproject_20260227_143025_a1b2"
    priority: TaskPriority = TaskPriority.NORMAL
    is_system_command: bool = False
    is_p2p: bool = False  # True when message originates from a private (p2p) chat
    sender_id: str = ""  # Feishu open_id of the message sender
    sender_union_id: str = ""  # Cross-app stable Feishu union_id of the sender
    tenant_key: str = ""  # Authoritative tenant from the Lark event header

    def get_effective_queue_key(self) -> str:
        """Calculate the effective queue key for routing.

        Routing rules:
        - System commands: {chat_id}:SYSTEM (high concurrency, bypasses per-key limit)
        - Project tasks: {chat_id}:{project_id} (serial within project)
        - No project: {chat_id}:DEFAULT (serial)
        """
        if self.queue_key:
            return self.queue_key
        if self.is_system_command:
            return f"{self.chat_id}{SYSTEM_QUEUE_SUFFIX}"
        if self.project_id:
            return f"{self.chat_id}:{self.project_id}"
        return f"{self.chat_id}{DEFAULT_QUEUE_SUFFIX}"

class TaskEvent(BaseModel):
    """任务状态/进度事件（用于 listeners 与可观测性输出）。"""
    model_config = ConfigDict(frozen=True)

    run_id: str
    chat_id: str
    status: TaskStatus
    timestamp: float
    name: str
    task_type: str
    project_id: Optional[str] = None
    message_id: Optional[str] = None
    origin_message_id: Optional[str] = None
    request_id: Optional[str] = None
    task_id: Optional[str] = None
    progress_message: Optional[str] = None
    progress_percent: Optional[float] = None
    error: Optional[str] = None


class CancellationToken:
    """任务取消令牌。

    约定：调度器不会强制中断运行中的线程；任务需要主动检查该 token。
    """

    def __init__(self):
        self._evt = threading.Event()

    def cancel(self):
        """标记为已取消（幂等）。"""
        self._evt.set()

    @property
    def is_canceled(self) -> bool:
        return self._evt.is_set()

    def raise_if_canceled(self):
        """若已取消则抛 `TaskCanceledError`。"""
        if self.is_canceled:
            raise TaskCanceledError("task canceled")


class TaskCanceledError(RuntimeError):
    """任务取消异常（任务内部可用来提前退出）。"""

    pass


class TaskQueueFullError(RateLimitExceededException):
    """Raised when a scheduler admission lane has reached its pending limit."""

    def __init__(self, lane: str, limit: int):
        self.lane = lane
        self.limit = limit
        super().__init__(f"Task scheduler {lane} queue is full (limit={limit})")


class TaskRunState(BaseModel):
    """任务运行态（调度器内部 SSOT）。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: TaskSpec
    run_id: str
    assigned_queue_key: str = ""
    project_serial_key: Optional[str] = None
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    progress_message: Optional[str] = None
    progress_percent: Optional[float] = None
    error: Optional[str] = None
    cancellation: CancellationToken = Field(default_factory=CancellationToken)


class TaskContext:
    """Context passed to the running task function."""

    def __init__(self, scheduler: "TaskScheduler", run_id: str, token: CancellationToken, spec: "TaskSpec"):
        self._scheduler = scheduler
        self.run_id = run_id
        self.cancel_token = token
        self.spec = spec

    def progress(self, message: str, percent: Optional[float] = None):
        """更新任务进度（仅对 RUNNING 有效）。"""
        self._scheduler.update_progress(self.run_id, message=message, percent=percent)

    def check_canceled(self):
        """若任务已取消则抛异常（建议任务函数周期性调用）。"""
        self.cancel_token.raise_if_canceled()


class _QueuedTask(BaseModel):
    """队列中的任务项（内部数据结构）。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    spec: TaskSpec
    fn: Callable[['TaskContext'], Any]
    context: contextvars.Context


class _TaskCompletion:
    """One-shot terminal event retained by the public task handle."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # leaf lock: never acquires scheduler state
        self._event: TaskEvent | None = None
        self._callbacks: list[Callable[[TaskEvent], None]] = []

    def add_done_callback(self, callback: Callable[[TaskEvent], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            event = self._event
            if event is None:
                self._callbacks.append(callback)
                return
        self._invoke(callback, event)

    def complete(self, event: TaskEvent) -> tuple[Callable[[TaskEvent], None], ...]:
        with self._lock:
            if self._event is not None:
                return ()
            self._event = event
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
            return callbacks

    @staticmethod
    def _invoke(callback: Callable[[TaskEvent], None], event: TaskEvent) -> None:
        try:
            callback(event)
        except BaseException:
            logger.debug("task completion callback raised an exception", exc_info=True)


class TaskHandle:
    """Non-blocking task identity and cancellation handle."""

    def __init__(
        self,
        scheduler: "TaskScheduler",
        run_id: str,
        completion: _TaskCompletion,
    ):
        self._scheduler = scheduler
        self.run_id = run_id
        self._completion = completion

    def cancel(self) -> bool:
        """请求取消任务。"""
        return self._scheduler.cancel(self.run_id)

    def add_done_callback(self, callback: Callable[[TaskEvent], None]) -> None:
        """Run *callback* once with the immutable terminal event.

        Registration is race-free with completion.  A late registration is
        replayed immediately and does not depend on scheduler history retention.
        Callback failures are isolated from task accounting.
        """

        self._completion.add_done_callback(callback)

class TaskScheduler:
    """Thread-safe scheduler with project serialization and a system fast lane."""

    _TRANSITIONS = {
        TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELED}),
        TaskStatus.RUNNING: frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}),
    }
    _TERMINAL_STATUSES = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED})
    _REAP_DEFAULT_MAX_AGE = 300.0
    _RUN_ID_GENERATION_ATTEMPTS = 16

    def __init__(
        self,
        *,
        max_concurrent: int = 10,
        per_key_concurrency: int = 1,
        system_concurrency: int = 10,
        max_pending_normal: int = 1000,
        max_pending_system: int = 100,
        max_terminal_history: int = 5000,
        worker_executor: Optional[ThreadPoolExecutor] = None,
        system_executor: Optional[ThreadPoolExecutor] = None,
        thread_name_prefix: str = "task_worker",
        run_guard: Optional[Callable[[], ContextManager[Any]]] = None,
        run_guard_cancel: Optional[Callable[[], None]] = None,
        run_guard_timeout_s: float | None = None,
    ):
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        if per_key_concurrency <= 0:
            raise ValueError("per_key_concurrency must be > 0")
        if system_concurrency <= 0:
            raise ValueError("system_concurrency must be > 0")
        if max_pending_normal <= 0:
            raise ValueError("max_pending_normal must be > 0")
        if max_pending_system <= 0:
            raise ValueError("max_pending_system must be > 0")
        if max_terminal_history <= 0:
            raise ValueError("max_terminal_history must be > 0")
        if run_guard_timeout_s is not None and (
            not isinstance(run_guard_timeout_s, (int, float))
            or isinstance(run_guard_timeout_s, bool)
            or not math.isfinite(run_guard_timeout_s)
            or run_guard_timeout_s <= 0
        ):
            raise ValueError("run_guard_timeout_s must be finite and > 0")
        self._max_concurrent = max_concurrent
        self._per_key = per_key_concurrency
        self._system_concurrency = system_concurrency
        self._max_pending_normal = max_pending_normal
        self._max_pending_system = max_pending_system
        self._max_terminal_history = max_terminal_history
        self._run_guard_timeout_s = (
            float(run_guard_timeout_s) if run_guard_timeout_s is not None else None
        )
        self._run_guard = run_guard or nullcontext
        guard_owner = getattr(run_guard, "__self__", None)
        cancellable_run_guard = getattr(
            guard_owner,
            "cancellable_task_guard",
            None,
        )
        self._cancellable_run_guard = (
            cancellable_run_guard if callable(cancellable_run_guard) else None
        )
        self._run_guard_cancel = run_guard_cancel or getattr(
            guard_owner,
            "cancel_waiters",
            None,
        )

        self._executor = worker_executor or ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix=thread_name_prefix,
        )
        self._system_executor = system_executor or ThreadPoolExecutor(
            max_workers=system_concurrency,
            thread_name_prefix=f"{thread_name_prefix}_system",
        )

        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._cv = threading.Condition(self._lock)
        self._queues: dict[str, Deque[_QueuedTask]] = defaultdict(deque)  # queue_key -> queue
        self._running_by_key: dict[str, int] = defaultdict(int)
        self._running_by_project: dict[str, int] = defaultdict(int)
        self._running_total_normal = 0
        self._running_total_system = 0
        self._pending_normal = 0
        self._pending_system = 0
        self._active_run_ids: set[str] = set()
        self._worker_futures: dict[str, Future[Any]] = {}
        self._worker_started: set[str] = set()
        self._states: dict[str, TaskRunState] = {}
        self._terminal_order: Deque[str] = deque()
        self._completions: dict[str, _TaskCompletion] = {}
        self._listeners: list[Callable[[TaskEvent], None]] = []
        self._listener_inflight: dict[int, int] = {}
        self._event_queue: queue.Queue[
            tuple[
                TaskEvent,
                tuple[Callable[[TaskEvent], None], ...],
            ]
            | None
        ] = queue.Queue()
        self._completion_queue: queue.Queue[
            tuple[TaskEvent, tuple[Callable[[TaskEvent], None], ...]] | None
        ] = queue.Queue()
        self._event_dispatch_stopping = False
        self._completion_dispatch_stopping = False
        self._event_dispatcher = threading.Thread(
            target=self._event_dispatch_loop,
            name="task_scheduler_events",
            daemon=True,
        )
        self._completion_dispatcher = threading.Thread(
            target=self._completion_dispatch_loop,
            name="task_scheduler_completions",
            daemon=True,
        )
        self._event_dispatcher.start()
        self._completion_dispatcher.start()

        self._rate_limiters: dict[str, RateLimiter] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._by_chat: dict[str, Deque[str]] = defaultdict(deque)
        self._by_project: dict[str, Deque[str]] = defaultdict(deque)
        self._by_task_id: dict[str, str] = {}
        self._admission_fenced = False
        self._stopped = False
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="task_scheduler_dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    def register_policy(
        self,
        task_type: str,
        rate_limiter: Optional[RateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """为指定 `task_type` 注册背压/熔断策略。"""
        with self._lock:
            if rate_limiter:
                self._rate_limiters[task_type] = rate_limiter
            if circuit_breaker:
                self._circuit_breakers[task_type] = circuit_breaker

    def add_listener(self, listener: Callable[[TaskEvent], None]):
        """添加任务事件监听器（best-effort，监听器异常会被吞掉）。"""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[TaskEvent], None]) -> bool:
        """Remove every match and quiesce callbacks already in progress."""

        with self._cv:
            removed_ids = {
                id(registered)
                for registered in self._listeners
                if registered == listener
            }
            retained = [registered for registered in self._listeners if registered != listener]
            if len(retained) == len(self._listeners):
                return False
            self._listeners = retained
            if threading.current_thread() is not self._event_dispatcher:
                self._cv.wait_for(
                    lambda: not any(
                        self._listener_inflight.get(listener_id, 0)
                        for listener_id in removed_ids
                    )
                )
            return True

    @staticmethod
    def _build_project_serial_key(chat_id: str, project_id: Optional[str]) -> Optional[str]:
        if not project_id:
            return None
        return f"{chat_id}:{project_id}"

    def _requeue_item_unlocked(self, queue_key: str, item: _QueuedTask) -> None:
        q = self._queues[queue_key]
        if item.spec.priority == TaskPriority.HIGH:
            q.appendleft(item)
        else:
            q.append(item)
        if self._is_system_queue(queue_key):
            self._pending_system += 1
        else:
            self._pending_normal += 1

    def _decrement_pending_unlocked(self, queue_key: str) -> None:
        if self._is_system_queue(queue_key):
            self._pending_system = max(0, self._pending_system - 1)
        else:
            self._pending_normal = max(0, self._pending_normal - 1)

    def _drain_queued_tasks_unlocked(self) -> None:
        for key, q in list(self._queues.items()):
            while q:
                item = q.popleft()
                self._decrement_pending_unlocked(key)
                self._transition_unlocked(item.run_id, TaskStatus.CANCELED)
            self._queues.pop(key, None)

    def _transition_unlocked(
        self,
        run_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> TaskRunState | None:
        """Apply one legal transition; terminal states fence all late writers."""
        state = self._states.get(run_id)
        if state is None or status not in self._TRANSITIONS.get(state.status, ()):
            return None
        state.status = status
        now = time.time()
        if status is TaskStatus.RUNNING:
            state.started_at = now
        elif status in self._TERMINAL_STATUSES:
            state.ended_at = now
            state.progress_message = None
            state.progress_percent = None
            self._terminal_order.append(run_id)
        state.error = error
        self._emit(state)
        if status in self._TERMINAL_STATUSES:
            self._reap_completed_states()
        return state

    def submit(self, spec: TaskSpec, fn: Callable[[TaskContext], Any]) -> TaskHandle:
        """提交任务（入队）并返回 `TaskHandle`。

        该方法会先执行 *task_type 级别* 的背压检查：
        - CircuitBreaker OPEN -> 直接拒绝
        - RateLimiter 获取失败 -> 直接拒绝

        通过后才会生成 `run_id`、落地 `TaskRunState`，并推入对应的队列。
        """
        key = spec.get_effective_queue_key()
        with self._cv:
            if self._stopped or self._admission_fenced:
                raise RuntimeError("TaskScheduler admission is fenced")

            is_system = self._is_system_queue(key)
            pending = self._pending_system if is_system else self._pending_normal
            pending_limit = self._max_pending_system if is_system else self._max_pending_normal
            if pending >= pending_limit:
                raise TaskQueueFullError("system" if is_system else "normal", pending_limit)

            rl = self._rate_limiters.get(spec.task_type)
            cb = self._circuit_breakers.get(spec.task_type)

            if cb and cb.state == CircuitState.OPEN:
                raise CircuitBreakerOpenException(f"Circuit breaker OPEN for task type {spec.task_type}")

            if rl and not rl.acquire(1, blocking=False):
                raise RateLimitExceededException(f"Rate limit exceeded for task type {spec.task_type}")

            for _attempt in range(self._RUN_ID_GENERATION_ATTEMPTS):
                run_id = uuid.uuid4().hex
                if run_id not in self._states:
                    break
            else:
                raise RuntimeError("TaskScheduler could not allocate a unique run_id")
            state = TaskRunState(
                spec=spec,
                run_id=run_id,
                assigned_queue_key=key,
                project_serial_key=self._build_project_serial_key(spec.chat_id, spec.project_id),
            )
            completion = _TaskCompletion()
            item = _QueuedTask(
                run_id=run_id,
                spec=spec,
                fn=fn,
                context=contextvars.copy_context(),
            )
            self._states[run_id] = state
            self._completions[run_id] = completion
            if spec.task_id:
                self._by_task_id[spec.task_id] = run_id
            self._by_chat[spec.chat_id].append(run_id)
            if spec.project_id:
                self._by_project[str(spec.project_id)].append(run_id)
            self._requeue_item_unlocked(key, item)

            self._emit(state)
            self._cv.notify_all()

        return TaskHandle(self, run_id, completion)

    def update_project_id(self, run_id: str, project_id: Optional[str]) -> bool:
        """Best-effort update of project_id for an existing task.

        Useful when the project is resolved inside the task body (e.g. by
        reply-chain mapping) rather than at submit time.
        """
        if not project_id:
            return False

        with self._lock:
            state = self._states.get(run_id)
            if not state:
                return False
            old_project = state.spec.project_id
            if old_project == project_id:
                return True

            new_spec = state.spec.model_copy(update={"project_id": str(project_id)})
            new_queue_key = new_spec.get_effective_queue_key()
            new_project_key = self._build_project_serial_key(new_spec.chat_id, new_spec.project_id)

            if state.status == TaskStatus.QUEUED and state.assigned_queue_key != new_queue_key:
                item = self._pop_queued_item_unlocked(run_id, state.assigned_queue_key)
                if item:
                    item = _QueuedTask(run_id=item.run_id, spec=new_spec, fn=item.fn, context=item.context)
                    self._requeue_item_unlocked(new_queue_key, item)
                    state.assigned_queue_key = new_queue_key
            elif state.status == TaskStatus.RUNNING and state.project_serial_key != new_project_key:
                if state.project_serial_key:
                    self._running_by_project[state.project_serial_key] = max(
                        0, self._running_by_project[state.project_serial_key] - 1
                    )
                    if self._running_by_project[state.project_serial_key] == 0:
                        self._running_by_project.pop(state.project_serial_key, None)
                if new_project_key:
                    self._running_by_project[new_project_key] += 1

            state.spec = new_spec
            state.project_serial_key = new_project_key

            if old_project:
                self._remove_from_index_unlocked(self._by_project, str(old_project), run_id)
            self._by_project[str(project_id)].append(run_id)
            self._emit(state)
            self._cv.notify_all()
            return True

    def cancel(self, run_id: str) -> bool:
        """取消任务。

        - 若任务仍在队列中：移除并标记为 `CANCELED`。
        - 若任务已在运行：仅设置取消令牌，任务需主动检查。
        """
        future_to_cancel: Future[Any] | None = None
        with self._cv:
            state = self._states.get(run_id)
            if not state:
                return False
            if state.status in self._TERMINAL_STATUSES:
                return False

            state.cancellation.cancel()

            if state.status == TaskStatus.QUEUED:
                self._remove_from_queue_unlocked(run_id, state.assigned_queue_key or state.spec.get_effective_queue_key())
                self._transition_unlocked(run_id, TaskStatus.CANCELED)
                self._cv.notify_all()
                return True

            if run_id not in self._worker_started:
                future_to_cancel = self._worker_futures.get(run_id)
            self._cv.notify_all()
        if future_to_cancel is not None:
            future_to_cancel.cancel()
        return True

    def update_progress(self, run_id: str, *, message: str, percent: Optional[float] = None):
        """更新任务进度（仅 RUNNING 状态有效）。"""
        with self._lock:
            state = self._states.get(run_id)
            if not state:
                return
            if state.status != TaskStatus.RUNNING:
                return
            state.progress_message = message
            if percent is not None:
                # clamp into [0, 100]
                state.progress_percent = max(0.0, min(100.0, float(percent)))
            self._emit(state)

    def get_state(self, run_id: str) -> Optional[TaskRunState]:
        """获取任务运行态（不存在返回 None）。"""
        with self._lock:
            return self._states.get(run_id)

    def has_running_task(
        self,
        chat_id: str,
        *,
        task_types: frozenset[str],
    ) -> bool:
        """Return authoritative RUNNING truth for selected task types."""

        if not task_types:
            return False
        with self._lock:
            return any(
                state is not None
                and state.status is TaskStatus.RUNNING
                and state.spec.task_type in task_types
                for run_id in self._by_chat.get(chat_id, ())
                if (state := self._states.get(run_id)) is not None
            )

    def get_state_by_task_id(self, task_id: str, chat_id: str) -> Optional[TaskRunState]:
        """Look up a task by its human-readable task_id.

        Supports exact match and partial suffix match (last 6+ chars).
        *chat_id* is required — only returns the task if it belongs to that chat
        (enforces cross-chat isolation at the API level).
        """
        with self._lock:
            run_id = self._by_task_id.get(task_id)
            if run_id is None and len(task_id) >= 4:
                matches = [rid for tid, rid in self._by_task_id.items() if tid.endswith(task_id) or task_id in tid]
                run_id = matches[0] if len(matches) == 1 else None
            state = self._states.get(run_id) if run_id else None
            if state and state.spec.chat_id != chat_id:
                return None
            return state

    def list_tasks(
        self,
        *,
        chat_id: Optional[str] = None,
        project_id: Optional[str] = None,
        include_done: bool = False,
        limit: int = 50,
    ) -> list[TaskRunState]:
        """Query tasks by chat/project.

        - If both chat_id and project_id are provided, it returns the intersection.
        - By default, it returns non-terminal tasks only.
        """
        if limit <= 0:
            return []

        with self._lock:
            if chat_id is not None and project_id is not None:
                proj_ids = set(self._by_project.get(str(project_id), []))
                run_ids = [rid for rid in self._by_chat.get(chat_id, ()) if rid in proj_ids]
            elif chat_id is not None:
                run_ids = list(self._by_chat.get(chat_id, []))
            elif project_id is not None:
                run_ids = list(self._by_project.get(str(project_id), []))
            else:
                run_ids = list(self._states.keys())

            states = (self._states.get(rid) for rid in reversed(run_ids))
            return [
                state
                for state in states
                if state is not None and (include_done or state.status not in self._TERMINAL_STATUSES)
            ][:limit]

    def stop(self, *, wait: bool = False, shutdown_executor: bool = False):
        """停止调度器（best-effort）。

        - `wait=True`：等待 dispatcher thread 退出（短超时）。
        - `shutdown_executor=True`：关闭线程池（不会强杀运行中的任务）。
        """
        self.fence_admission()
        with self._cv:
            self._stopped = True
            self._maybe_stop_event_dispatcher_unlocked()
            self._cv.notify_all()
        if wait or shutdown_executor:
            self._dispatcher.join(timeout=2)
        if shutdown_executor:
            for executor in (self._executor, self._system_executor):
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    logger.debug("failed to shutdown executor", exc_info=True)
        if wait and self._event_dispatch_stopping:
            self._event_dispatcher.join(timeout=2)
        if wait and self._completion_dispatch_stopping:
            self._completion_dispatcher.join(timeout=2)

    def fence_admission(self) -> None:
        """Reject new work and cancel work that has not entered its callback.

        The cross-process restart writer may already be holding the admission
        lock.  Canceling local guard waiters lets those reserved worker slots
        converge to ``CANCELED`` instead of remaining falsely ``RUNNING`` while
        shutdown proceeds.
        """

        futures_to_cancel: list[Future[Any]] = []
        with self._cv:
            if not self._admission_fenced:
                self._admission_fenced = True
                self._drain_queued_tasks_unlocked()
                futures_to_cancel = [
                    future
                    for run_id, future in self._worker_futures.items()
                    if run_id not in self._worker_started
                ]
                self._cv.notify_all()
        for future in futures_to_cancel:
            future.cancel()
        if callable(self._run_guard_cancel):
            try:
                self._run_guard_cancel()
            except Exception:
                logger.exception("failed to cancel run-guard waiters")

    def cancel_active(self) -> int:
        """Request cooperative cancellation for every non-terminal worker."""

        canceled = 0
        futures_to_cancel: list[Future[Any]] = []
        with self._cv:
            for state in self._states.values():
                if state.status != TaskStatus.RUNNING:
                    continue
                if not state.cancellation.is_canceled:
                    state.cancellation.cancel()
                    canceled += 1
                if state.run_id not in self._worker_started:
                    future = self._worker_futures.get(state.run_id)
                    if future is not None:
                        futures_to_cancel.append(future)
            self._cv.notify_all()
        for future in futures_to_cancel:
            future.cancel()
        return canceled

    def wait_for_idle(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for all reserved worker slots to drain."""

        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._running_total_normal or self._running_total_system:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def _reap_completed_states(self, max_age_seconds: float = _REAP_DEFAULT_MAX_AGE) -> int:
        cutoff = time.time() - max_age_seconds
        removed = 0

        while self._terminal_order:
            run_id = self._terminal_order[0]
            state = self._states.get(run_id)
            if state is None:
                self._terminal_order.popleft()
                continue
            if run_id in self._active_run_ids:
                break
            if (state.ended_at or state.created_at) > cutoff:
                break
            self._terminal_order.popleft()
            self._remove_state_unlocked(run_id, state)
            removed += 1

        terminal_count = len(self._terminal_order)
        while terminal_count > self._max_terminal_history and self._terminal_order:
            run_id = self._terminal_order[0]
            state = self._states.get(run_id)
            if state is None:
                self._terminal_order.popleft()
                terminal_count -= 1
                continue
            if run_id in self._active_run_ids:
                break
            self._terminal_order.popleft()
            self._remove_state_unlocked(run_id, state)
            terminal_count -= 1
            removed += 1
        return removed

    def _remove_state_unlocked(self, run_id: str, state: TaskRunState) -> None:
        self._states.pop(run_id, None)
        self._remove_from_index_unlocked(self._by_chat, state.spec.chat_id, run_id)
        if state.spec.project_id:
            self._remove_from_index_unlocked(self._by_project, str(state.spec.project_id), run_id)
        if state.spec.task_id and self._by_task_id.get(state.spec.task_id) == run_id:
            self._by_task_id.pop(state.spec.task_id, None)

    def _dispatch_loop(self):
        """调度循环：不断从队列中挑选可运行任务并投递到线程池。

        该循环运行在一个 daemon thread 中：
        - 等待条件：有任务可跑 / 有任务结束释放并发额度 / stop() 通知
        - 选择策略：优先 SYSTEM 队列（高并发），否则按队列公平选择（尽量避免饥饿）
        """
        while True:
            future_to_observe: Future[Any] | None = None
            observed_task: _QueuedTask | None = None
            with self._cv:
                if self._stopped:
                    return

                task = self._pick_next_task_unlocked()
                if not task:
                    self._reap_completed_states()
                    self._cv.wait(timeout=0.2)
                    continue

                state = self._states.get(task.run_id)
                key = state.assigned_queue_key if state and state.assigned_queue_key else task.spec.get_effective_queue_key()
                is_system = self._is_system_queue(key)
                if is_system:
                    self._running_total_system += 1
                else:
                    self._running_total_normal += 1
                self._running_by_key[key] += 1
                if state and state.project_serial_key:
                    self._running_by_project[state.project_serial_key] += 1
                if state:
                    self._transition_unlocked(task.run_id, TaskStatus.RUNNING)
                self._active_run_ids.add(task.run_id)
                self._cv.notify_all()

                try:
                    executor = self._system_executor if is_system else self._executor
                    fut = executor.submit(self._run_wrapper, task)
                except Exception as e:
                    # rollback running counters and converge to terminal state
                    if state:
                        self._transition_unlocked(task.run_id, TaskStatus.FAILED, error=get_error_detail(e))
                    self._release_running_slot_unlocked(task)
                    self._cv.notify_all()
                    continue
                self._worker_futures[task.run_id] = fut
                future_to_observe = fut
                observed_task = task
            if future_to_observe is not None and observed_task is not None:
                future_to_observe.add_done_callback(
                    lambda future, run_id=observed_task.run_id, task=observed_task: (
                        self._on_worker_future_done(run_id, task, future)
                    )
                )

    def _is_system_queue(self, key: str) -> bool:
        """判断队列 key 是否为系统快通道。"""
        return key.endswith(SYSTEM_QUEUE_SUFFIX)

    def _get_key_concurrency_limit(self, key: str) -> int:
        """返回指定 queue_key 的并发上限。"""
        if self._is_system_queue(key):
            return self._system_concurrency
        return self._per_key

    def _pick_next_task_unlocked(self) -> Optional[_QueuedTask]:
        """挑选一个当前可运行的任务（调用方需持有 `_cv` 锁）。"""
        for key, q in list(self._queues.items()):
            if not q:
                self._queues.pop(key, None)
                continue

            # Global capacity gate:
            # - System queue: reserved executor (does not consume normal capacity)
            # - Normal queues: bounded by max_concurrent
            if self._is_system_queue(key):
                if self._running_total_system >= self._system_concurrency:
                    continue
            else:
                if self._running_total_normal >= self._max_concurrent:
                    continue
            limit = self._get_key_concurrency_limit(key)
            if self._running_by_key.get(key, 0) >= limit:
                continue

            while q:
                item = q[0]
                st = self._states.get(item.run_id)
                if not st or st.status is not TaskStatus.QUEUED:
                    q.popleft()
                    self._decrement_pending_unlocked(key)
                    continue
                if st.cancellation.is_canceled:
                    q.popleft()
                    self._decrement_pending_unlocked(key)
                    self._transition_unlocked(item.run_id, TaskStatus.CANCELED)
                    continue
                if st and st.project_serial_key and self._running_by_project.get(st.project_serial_key, 0) >= 1:
                    # Project-level serialization: keep per-project tasks ordered.
                    # Exception: control-plane tasks (/help, /exit) must not be blocked.
                    if st.spec.task_type not in {"system_help", "system_exit"}:
                        # keep queue order; try other queues first
                        break
                break

            if not q:
                self._queues.pop(key, None)
                continue

            item = q.popleft()
            self._decrement_pending_unlocked(key)
            if not q:
                self._queues.pop(key, None)
            return item

        return None

    def _run_wrapper(self, task: _QueuedTask):
        """Wrapper to run the task in its captured context."""
        with self._cv:
            if task.run_id not in self._active_run_ids:
                return None
            self._worker_started.add(task.run_id)
        return task.context.run(self._run_in_task_context, task)

    def _on_worker_future_done(
        self,
        run_id: str,
        task: _QueuedTask,
        future: Future[Any],
    ) -> None:
        if not future.cancelled():
            return
        with self._cv:
            if self._worker_futures.get(run_id) is not future:
                return
            self._worker_futures.pop(run_id, None)
            if run_id in self._worker_started:
                return
            state = self._states.get(run_id)
            if state is not None:
                self._transition_unlocked(run_id, TaskStatus.CANCELED)
            self._release_running_slot_unlocked(task)
            self._cv.notify_all()

    def _run_in_task_context(self, task: _QueuedTask):
        """Publish this run id while executing guard, callback, and terminal work."""
        token = _CURRENT_TASK_RUN_ID.set(task.run_id)
        try:
            return self._run_with_guard(task)
        finally:
            _CURRENT_TASK_RUN_ID.reset(token)

    def _run_with_guard(self, task: _QueuedTask):
        """Keep the process-wide run guard around callback + terminal accounting.

        The callback outcome is the scheduler's single terminal truth.  A guard
        release error is logged after the guard has had its close opportunity,
        but never rewrites or re-emits an already-published terminal state.
        """

        with self._cv:
            state = self._states.get(task.run_id)
            cancellation = state.cancellation if state is not None else CancellationToken()
            if state is None:
                cancellation.cancel()

        def canceled_before_callback() -> bool:
            return cancellation.is_canceled or self._admission_fenced

        try:
            if self._cancellable_run_guard is None:
                guard = self._run_guard()
            else:
                deadline = (
                    None
                    if self._run_guard_timeout_s is None
                    else time.monotonic() + self._run_guard_timeout_s
                )
                guard = self._cancellable_run_guard(
                    canceled=canceled_before_callback,
                    deadline=deadline,
                )
            guard.__enter__()
        except BaseException as exc:
            with self._cv:
                state = self._states.get(task.run_id)
                # The dispatcher reserved a running slot before the worker
                # attempted admission.  A fail-closed admission must converge
                # that reservation exactly once.
                if state:
                    canceled = (
                        self._admission_fenced
                        or state.cancellation.is_canceled
                    )
                    self._transition_unlocked(
                        task.run_id,
                        TaskStatus.CANCELED if canceled else TaskStatus.FAILED,
                        error=None if canceled else get_error_detail(exc),
                    )
                self._release_running_slot_unlocked(task)
                self._cv.notify_all()
            raise

        # This locked check is the callback-start linearization point.  A
        # cancellation or admission fence that wins the condition lock keeps
        # the callback from starting even when the cross-process locks were
        # acquired immediately beforehand.
        with self._cv:
            state = self._states.get(task.run_id)
            callback_rejected = bool(
                state is None
                or state.status is not TaskStatus.RUNNING
                or state.cancellation.is_canceled
                or self._admission_fenced
            )
            if callback_rejected:
                if state is not None and state.status is TaskStatus.RUNNING:
                    self._transition_unlocked(task.run_id, TaskStatus.CANCELED)
                if task.run_id in self._active_run_ids:
                    self._release_running_slot_unlocked(task)
                self._cv.notify_all()

        if callback_rejected:
            canceled = TaskCanceledError("task canceled before callback")
            try:
                guard.__exit__(type(canceled), canceled, canceled.__traceback__)
            except BaseException:
                logger.exception(
                    "run guard release failed after pre-callback cancellation "
                    "run_id=%s",
                    task.run_id,
                )
            return None

        try:
            value = self._do_run(task)
        except BaseException:
            exc_type, exc, traceback = sys.exc_info()
            try:
                suppressed = guard.__exit__(exc_type, exc, traceback)
            except BaseException:
                logger.exception(
                    "run guard release failed after task terminal accounting "
                    "run_id=%s",
                    task.run_id,
                )
                suppressed = False
            if suppressed:
                return None
            raise

        try:
            guard.__exit__(None, None, None)
        except BaseException:
            # RestartGate.task_guard always attempts close(2), which releases
            # flock ownership.  Treat a release error as an operational alert:
            # changing SUCCEEDED to FAILED here would expose two contradictory
            # terminal events to callers.
            logger.exception(
                "run guard release failed after task terminal accounting "
                "run_id=%s",
                task.run_id,
            )
        return value

    def _release_running_slot_unlocked(self, task: _QueuedTask) -> None:
        if task.run_id not in self._active_run_ids:
            return
        self._worker_futures.pop(task.run_id, None)
        self._worker_started.discard(task.run_id)
        state = self._states.get(task.run_id)
        key = (
            state.assigned_queue_key
            if state and state.assigned_queue_key
            else task.spec.get_effective_queue_key()
        )
        if self._is_system_queue(key):
            self._running_total_system = max(0, self._running_total_system - 1)
        else:
            self._running_total_normal = max(0, self._running_total_normal - 1)
        remaining_by_key = max(0, self._running_by_key.get(key, 0) - 1)
        if remaining_by_key:
            self._running_by_key[key] = remaining_by_key
        else:
            self._running_by_key.pop(key, None)
        if state and state.project_serial_key:
            remaining_by_project = max(
                0,
                self._running_by_project.get(state.project_serial_key, 0) - 1,
            )
            if remaining_by_project:
                self._running_by_project[state.project_serial_key] = remaining_by_project
            else:
                self._running_by_project.pop(state.project_serial_key, None)
        self._active_run_ids.discard(task.run_id)
        self._reap_completed_states()
        self._maybe_stop_event_dispatcher_unlocked()

    def _do_run(self, task: _QueuedTask):
        """执行任务并维护状态（运行在 worker thread 中）。"""
        run_id = task.run_id
        spec = task.spec
        state = self.get_state(run_id)
        token = state.cancellation if state else CancellationToken()
        ctx = TaskContext(self, run_id=run_id, token=token, spec=spec)

        try:
            token.raise_if_canceled()

            cb = self._circuit_breakers.get(spec.task_type)
            if cb:
                value = cb.call(task.fn, ctx)
            else:
                value = task.fn(ctx)

            token.raise_if_canceled()

            with self._cv:
                st = self._states.get(run_id)
                if st:
                    self._transition_unlocked(run_id, TaskStatus.SUCCEEDED)
                self._cv.notify_all()
                return value

        except TaskCanceledError:
            with self._cv:
                st = self._states.get(run_id)
                if st:
                    self._transition_unlocked(run_id, TaskStatus.CANCELED)
                self._cv.notify_all()
            return None

        except BaseException as e:
            with self._cv:
                st = self._states.get(run_id)
                if st:
                    self._transition_unlocked(run_id, TaskStatus.FAILED, error=get_error_detail(e))
                self._cv.notify_all()
            raise

        finally:
            with self._cv:
                self._release_running_slot_unlocked(task)
                self._cv.notify_all()

    def _emit(self, st: TaskRunState):
        """Queue an immutable event; user callbacks always run outside locks."""
        ev = TaskEvent(
            run_id=st.run_id,
            chat_id=st.spec.chat_id,
            status=st.status,
            timestamp=time.time(),
            name=st.spec.name,
            task_type=st.spec.task_type,
            project_id=st.spec.project_id,
            message_id=st.spec.message_id,
            origin_message_id=st.spec.origin_message_id,
            request_id=st.spec.request_id,
            task_id=st.spec.task_id,
            progress_message=st.progress_message,
            progress_percent=st.progress_percent,
            error=st.error,
        )
        completion_callbacks: tuple[Callable[[TaskEvent], None], ...] = ()
        if st.status in self._TERMINAL_STATUSES:
            completion = self._completions.pop(st.run_id, None)
            if completion is not None:
                completion_callbacks = completion.complete(ev)
        self._event_queue.put((ev, tuple(self._listeners)))
        if completion_callbacks:
            self._completion_queue.put((ev, completion_callbacks))

    def _event_dispatch_loop(self) -> None:
        while True:
            item = self._event_queue.get()
            if item is None:
                return
            event, listeners = item
            for listener in listeners:
                listener_id = id(listener)
                with self._cv:
                    if listener not in self._listeners:
                        continue
                    self._listener_inflight[listener_id] = (
                        self._listener_inflight.get(listener_id, 0) + 1
                    )
                try:
                    listener(event)
                except BaseException:
                    # listeners must never break scheduler
                    logger.debug("event listener raised an exception", exc_info=True)
                finally:
                    with self._cv:
                        remaining = self._listener_inflight.get(listener_id, 0) - 1
                        if remaining > 0:
                            self._listener_inflight[listener_id] = remaining
                        else:
                            self._listener_inflight.pop(listener_id, None)
                        self._cv.notify_all()

    def _completion_dispatch_loop(self) -> None:
        while True:
            item = self._completion_queue.get()
            if item is None:
                return
            event, callbacks = item
            for callback in callbacks:
                _TaskCompletion._invoke(callback, event)

    def _maybe_stop_event_dispatcher_unlocked(self) -> None:
        if (
            self._stopped
            and not self._active_run_ids
            and not self._event_dispatch_stopping
        ):
            self._event_dispatch_stopping = True
            self._event_queue.put(None)
        if (
            self._stopped
            and not self._active_run_ids
            and not self._completion_dispatch_stopping
        ):
            self._completion_dispatch_stopping = True
            self._completion_queue.put(None)

    def _pop_queued_item_unlocked(self, run_id: str, key: str) -> Optional[_QueuedTask]:
        q = self._queues.get(key)
        if not q:
            return None
        new_q: Deque[_QueuedTask] = deque()
        found: Optional[_QueuedTask] = None
        for item in q:
            if found is None and item.run_id == run_id:
                found = item
                continue
            new_q.append(item)
        if found is not None:
            self._decrement_pending_unlocked(key)
        if new_q:
            self._queues[key] = new_q
        else:
            self._queues.pop(key, None)
        return found

    def _remove_from_queue_unlocked(self, run_id: str, key: str):
        """从某个队列中移除指定 run_id（调用方需持锁）。"""
        self._pop_queued_item_unlocked(run_id, key)

    def _remove_from_index_unlocked(self, index: dict[str, Deque[str]], key: str, run_id: str):
        q = index.get(key)
        if not q:
            return
        index[key] = deque(rid for rid in q if rid != run_id)
        if not index[key]:
            index.pop(key, None)
