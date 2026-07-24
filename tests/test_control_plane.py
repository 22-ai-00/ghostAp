"""Tests for src.feishu.control_plane module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.feishu.control_plane import ControlPlane
from src.tasking import TaskEvent, TaskStatus


def _event(
    run_id: str,
    status: TaskStatus,
    *,
    chat_id: str = "chat1",
    task_type: str = "system_help",
) -> TaskEvent:
    return TaskEvent(
        run_id=run_id,
        chat_id=chat_id,
        status=status,
        timestamp=1.0,
        name="process_message",
        task_type=task_type,
    )


class TestControlPlane:
    """ControlPlane unit tests — verifies deferred exit and system command gating."""

    def _make_cp(self, *, scheduler=None, project_manager=None, exit_fn=None):
        scheduler = scheduler or MagicMock()
        pm = project_manager or MagicMock()
        fn = exit_fn or MagicMock()
        cp = ControlPlane(scheduler, pm, fn)
        return cp

    def test_is_system_cmd_inflight_initially_false(self):
        cp = self._make_cp()
        try:
            assert cp.is_system_cmd_inflight("chat1") is False
        finally:
            cp.stop()

    def test_system_cmd_gate_tracks_running(self):
        cp = self._make_cp()
        try:
            cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
            assert cp.is_system_cmd_inflight("chat1") is True
        finally:
            cp.stop()

    def test_system_cmd_gate_clears_on_success(self):
        cp = self._make_cp()
        try:
            cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
            cp.on_scheduler_event(_event("run-1", TaskStatus.SUCCEEDED))
            assert cp.is_system_cmd_inflight("chat1") is False
        finally:
            cp.stop()

    @pytest.mark.parametrize(
        "terminal_status",
        [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED],
    )
    def test_same_run_repeated_running_is_idempotent(self, terminal_status):
        cp = self._make_cp()
        try:
            cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
            cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
            cp.on_scheduler_event(_event("run-1", terminal_status))
            assert cp.is_system_cmd_inflight("chat1") is False
        finally:
            cp.stop()

    def test_distinct_runs_release_independently(self):
        cp = self._make_cp()
        try:
            cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
            cp.on_scheduler_event(_event("run-2", TaskStatus.RUNNING))
            cp.on_scheduler_event(_event("run-1", TaskStatus.SUCCEEDED))
            assert cp.is_system_cmd_inflight("chat1") is True
            cp.on_scheduler_event(_event("run-2", TaskStatus.FAILED))
            assert cp.is_system_cmd_inflight("chat1") is False
        finally:
            cp.stop()

    def test_system_command_gate_is_scoped_by_chat(self):
        cp = self._make_cp()
        try:
            cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
            cp.on_scheduler_event(_event("run-2", TaskStatus.RUNNING, chat_id="chat2"))
            cp.on_scheduler_event(_event("run-1", TaskStatus.CANCELED))
            assert cp.is_system_cmd_inflight("chat1") is False
            assert cp.is_system_cmd_inflight("chat2") is True
        finally:
            cp.stop()

    def test_request_deferred_exit_and_should_defer(self):
        scheduler = MagicMock()
        scheduler.list_tasks.return_value = []
        cp = self._make_cp(scheduler=scheduler)
        try:
            assert cp.should_defer_exit(chat_id="c1", project_id=None) is False
        finally:
            cp.stop()

    def test_stop_is_idempotent(self):
        cp = self._make_cp()
        cp.stop()
        cp.stop()  # should not raise
