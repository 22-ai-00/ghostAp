from __future__ import annotations

import threading
import time

import pytest

from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.team import coordinator as team_coordinator
from src.autonomous.team.coordinator import TeamCoordinatorActor, TeamCoordinatorError
from tests.autonomous.team_helpers import ImmediateTeamBackend, make_team_storage


class _BlockingResultBackend(ImmediateTeamBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._entered_count = 0
        self._entered_lock = threading.Lock()

    def result(self, acceptance_id):
        with self._entered_lock:
            self._entered_count += 1
            if self._entered_count >= 2:
                self.entered.set()
        assert self.release.wait(2.0)
        return super().result(acceptance_id)


def _start(actor: TeamCoordinatorActor, message_id: str):
    return actor.start_task(
        tenant_key="tenant_1",
        message_id=message_id,
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="implement and review the requested change",
    )


def test_team_close_is_bounded_and_retryable_while_backend_result_is_blocked(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = _BlockingResultBackend()
    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
    )
    first = _start(actor, "om_blocked_close_1")
    second = _start(actor, "om_blocked_close_2")
    queued = _start(actor, "om_queued_close")
    assert backend.entered.wait(1.0)

    started = time.monotonic()
    with pytest.raises(TeamCoordinatorError) as raised:
        actor.close(timeout_seconds=0.02)
    assert type(raised.value).__name__ == "TeamCoordinatorCloseTimeout"
    assert time.monotonic() - started < 0.2
    assert queued.run_id not in actor._active  # noqa: SLF001
    assert {first.run_id, second.run_id} <= actor._active  # noqa: SLF001
    with pytest.raises(TeamCoordinatorError, match="closed"):
        _start(actor, "om_after_close")

    backend.release.set()
    actor.close(timeout_seconds=1.0)
    assert actor._active == set()  # noqa: SLF001
    blobs.close()
    writer.close()


def test_runtime_close_preserves_shared_resources_when_team_does_not_stop() -> None:
    service_closed = False

    class _Team:
        @staticmethod
        def close(*, timeout_seconds: float) -> None:
            assert timeout_seconds > 0
            error_type = team_coordinator.TeamCoordinatorCloseTimeout
            raise error_type("team coordinator did not stop")

    class _Service:
        @staticmethod
        def stop_admission() -> None:
            return None

        @staticmethod
        def close() -> None:
            nonlocal service_closed
            service_closed = True

    runtime = EmployeeDepartmentRuntime()
    runtime._team = _Team()  # type: ignore[assignment]  # noqa: SLF001
    runtime._service = _Service()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="team_service:TeamCoordinatorCloseTimeout"):
        runtime.close()

    assert service_closed is False
    assert runtime._close_incomplete is True  # noqa: SLF001


def test_team_close_deadline_also_bounds_decision_provider_cleanup(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    release = threading.Event()
    entered = threading.Event()
    close_calls = 0

    class _DecisionProvider:
        def __call__(self, *_args):
            raise AssertionError("no task should invoke the provider")

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            entered.set()
            assert release.wait(2.0)

    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=ImmediateTeamBackend(),
        decision_provider=_DecisionProvider(),
    )
    watchdog = threading.Timer(0.3, release.set)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    try:
        with pytest.raises(TeamCoordinatorError) as raised:
            actor.close(timeout_seconds=0.02)
        assert type(raised.value).__name__ == "TeamCoordinatorCloseTimeout"
        assert time.monotonic() - started < 0.2
        assert entered.wait(0.1)
        assert close_calls == 1
    finally:
        release.set()
        watchdog.cancel()

    actor.close(timeout_seconds=1.0)
    assert close_calls == 1
    blobs.close()
    writer.close()
