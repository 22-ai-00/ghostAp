"""Integration contracts between TaskScheduler and the restart admission gate."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tasking.scheduler import TaskScheduler, TaskSpec, TaskStatus
from src.utils.restart_gate import RestartGate


def _ready_gate(tmp_path: Path) -> RestartGate:
    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())
    gate.mark_ready(service_pid=os.getpid())
    return gate


def _wait_for_status(
    scheduler: TaskScheduler,
    run_id: str,
    expected: TaskStatus,
    *,
    timeout: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = scheduler.get_state(run_id)
        if state is not None and state.status is expected:
            return
        time.sleep(0.005)
    state = scheduler.get_state(run_id)
    raise AssertionError(
        f"run {run_id} did not reach {expected}; current={getattr(state, 'status', None)}"
    )


def test_handle_cancel_interrupts_only_its_restart_guard_wait(
    tmp_path: Path,
) -> None:
    gate = _ready_gate(tmp_path)
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=gate.task_guard,
        run_guard_timeout_s=2,
    )
    callback_calls: list[str] = []

    try:
        with gate._lock_pair(exclusive=True, deadline=time.monotonic() + 2):
            canceled = scheduler.submit(
                TaskSpec(chat_id="chat", name="canceled-at-gate"),
                lambda _ctx: callback_calls.append("canceled"),
            )
            _wait_for_status(scheduler, canceled.run_id, TaskStatus.RUNNING)

            assert canceled.cancel() is True
            _wait_for_status(scheduler, canceled.run_id, TaskStatus.CANCELED)
            assert scheduler.wait_for_idle(timeout=0.5) is True
            assert callback_calls == []

        successor = scheduler.submit(
            TaskSpec(chat_id="chat", name="successor"),
            lambda _ctx: callback_calls.append("successor"),
        )
        _wait_for_status(scheduler, successor.run_id, TaskStatus.SUCCEEDED)
        assert callback_calls == ["successor"]
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_admission_fence_interrupts_restart_guard_without_running_callback(
    tmp_path: Path,
) -> None:
    gate = _ready_gate(tmp_path)
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=gate.task_guard,
        run_guard_timeout_s=2,
    )
    callback_calls: list[str] = []

    try:
        with gate._lock_pair(exclusive=True, deadline=time.monotonic() + 2):
            handle = scheduler.submit(
                TaskSpec(chat_id="chat", name="fenced-at-gate"),
                lambda _ctx: callback_calls.append("fenced"),
            )
            _wait_for_status(scheduler, handle.run_id, TaskStatus.RUNNING)

            scheduler.fence_admission()
            _wait_for_status(scheduler, handle.run_id, TaskStatus.CANCELED)
            assert scheduler.wait_for_idle(timeout=0.5) is True
            assert callback_calls == []
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)

    replacement = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=gate.task_guard,
        run_guard_timeout_s=2,
    )
    try:
        successor = replacement.submit(
            TaskSpec(chat_id="chat", name="post-fence-generation"),
            lambda _ctx: callback_calls.append("successor"),
        )
        _wait_for_status(replacement, successor.run_id, TaskStatus.SUCCEEDED)
        assert callback_calls == ["successor"]
    finally:
        replacement.stop(wait=True, shutdown_executor=True)


def test_fence_after_guard_acquire_still_prevents_callback_start(
    tmp_path: Path,
) -> None:
    gate = _ready_gate(tmp_path)
    guard_acquired = threading.Event()
    release_guard_enter = threading.Event()

    class PausingGate:
        @contextmanager
        def task_guard(self):
            with gate.task_guard():
                guard_acquired.set()
                assert release_guard_enter.wait(timeout=2)
                yield

        @contextmanager
        def cancellable_task_guard(self, *, canceled, deadline):
            with gate.cancellable_task_guard(
                canceled=canceled,
                deadline=deadline,
            ):
                guard_acquired.set()
                assert release_guard_enter.wait(timeout=2)
                yield

    pausing_gate = PausingGate()
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=pausing_gate.task_guard,
        run_guard_timeout_s=2,
    )
    callback_calls: list[str] = []

    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="chat", name="post-acquire-fence"),
            lambda _ctx: callback_calls.append("unexpected"),
        )
        assert guard_acquired.wait(timeout=1)

        scheduler.fence_admission()
        release_guard_enter.set()

        _wait_for_status(scheduler, handle.run_id, TaskStatus.CANCELED)
        assert scheduler.wait_for_idle(timeout=0.5) is True
        assert callback_calls == []
    finally:
        release_guard_enter.set()
        scheduler.stop(wait=True, shutdown_executor=True)


def test_restart_guard_deadline_fails_bounded_and_releases_worker_slot(
    tmp_path: Path,
) -> None:
    gate = _ready_gate(tmp_path)
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        run_guard=gate.task_guard,
        run_guard_timeout_s=0.15,
    )
    callback_calls: list[str] = []
    started = time.monotonic()

    try:
        with gate._lock_pair(exclusive=True, deadline=started + 2):
            handle = scheduler.submit(
                TaskSpec(chat_id="chat", name="guard-deadline"),
                lambda _ctx: callback_calls.append("expired"),
            )
            _wait_for_status(
                scheduler,
                handle.run_id,
                TaskStatus.FAILED,
                timeout=1,
            )
            state = scheduler.get_state(handle.run_id)
            assert state is not None
            assert state.error is not None
            assert "restart gate deadline expired" in state.error
            assert scheduler.wait_for_idle(timeout=0.5) is True
            assert callback_calls == []

        assert time.monotonic() - started < 1.0

        successor = scheduler.submit(
            TaskSpec(chat_id="chat", name="post-deadline"),
            lambda _ctx: callback_calls.append("successor"),
        )
        _wait_for_status(scheduler, successor.run_id, TaskStatus.SUCCEEDED)
        assert callback_calls == ["successor"]
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_production_scheduler_uses_configured_restart_deadline(
    tmp_path: Path,
) -> None:
    from src.feishu.ws_client import _build_task_scheduler

    class FakeGate:
        @contextmanager
        def task_guard(self):
            yield

        @contextmanager
        def cancellable_task_guard(self, *, canceled, deadline):
            del canceled, deadline
            yield

    settings = SimpleNamespace(
        restart_gate_dir="",
        restart_gate_timeout=17.5,
        task_scheduler_max_concurrent=1,
        task_scheduler_per_key_concurrency=1,
        system_command_concurrency=1,
        task_scheduler_max_pending_normal=10,
        task_scheduler_max_pending_system=2,
        task_scheduler_max_terminal_history=20,
    )

    with patch(
        "src.feishu.ws_client.RestartGate.for_project",
        return_value=FakeGate(),
    ):
        scheduler = _build_task_scheduler(settings, project_dir=tmp_path)
    try:
        assert scheduler._run_guard_timeout_s == 17.5
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


@pytest.mark.parametrize("unsafe", [0, -1, float("inf"), float("nan"), True])
def test_scheduler_rejects_invalid_run_guard_timeout(unsafe: object) -> None:
    scheduler = None
    try:
        with pytest.raises(ValueError, match="run_guard_timeout_s"):
            scheduler = TaskScheduler(run_guard_timeout_s=unsafe)
    finally:
        if scheduler is not None:
            scheduler.stop(wait=True, shutdown_executor=True)
