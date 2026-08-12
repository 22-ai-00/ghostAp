from __future__ import annotations

import threading

import pytest

from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime


class _CloseResource:
    def __init__(self, close_action) -> None:
        self.close_calls = 0
        self._close_action = close_action

    def close(self) -> None:
        self.close_calls += 1
        self._close_action()


def _start_close(
    runtime: EmployeeDepartmentRuntime,
) -> tuple[threading.Thread, threading.Event, list[BaseException]]:
    finished = threading.Event()
    errors: list[BaseException] = []

    def close() -> None:
        try:
            runtime.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=close)
    thread.start()
    return thread, finished, errors


def test_concurrent_close_waiters_observe_same_failed_attempt_then_retry() -> None:
    runtime = EmployeeDepartmentRuntime()
    entered = threading.Event()
    release = threading.Event()
    close_calls = 0

    def close_resource() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            entered.set()
            assert release.wait(2.0)
            raise RuntimeError("first resource close failed")

    resource = _CloseResource(close_resource)
    runtime._channels = resource  # type: ignore[assignment]  # noqa: SLF001

    owner, owner_finished, owner_errors = _start_close(runtime)
    assert entered.wait(1.0)
    waiter, waiter_finished, waiter_errors = _start_close(runtime)
    try:
        assert waiter_finished.wait(0.05) is False
    finally:
        release.set()
        owner.join(timeout=1.0)
        waiter.join(timeout=1.0)

    assert owner_finished.is_set()
    assert waiter_finished.is_set()
    assert len(owner_errors) == 1
    assert waiter_errors == owner_errors
    assert "channels:RuntimeError" in str(owner_errors[0])
    assert close_calls == 1
    assert resource.close_calls == 1

    runtime.close()

    assert close_calls == 2
    assert resource.close_calls == 2


def test_concurrent_close_waiter_preserves_owner_timeout_failure() -> None:
    runtime = EmployeeDepartmentRuntime()
    entered = threading.Event()
    release = threading.Event()

    def close_once() -> None:
        entered.set()
        assert release.wait(2.0)
        raise TimeoutError("owner close operation timed out")

    runtime._close_once = close_once  # type: ignore[method-assign]  # noqa: SLF001

    owner, owner_finished, owner_errors = _start_close(runtime)
    assert entered.wait(1.0)
    waiter, waiter_finished, waiter_errors = _start_close(runtime)
    try:
        assert waiter_finished.wait(0.05) is False
    finally:
        release.set()
        owner.join(timeout=1.0)
        waiter.join(timeout=1.0)

    assert owner_finished.is_set()
    assert waiter_finished.is_set()
    assert len(owner_errors) == 1
    assert waiter_errors == owner_errors
    assert type(waiter_errors[0]) is TimeoutError
    assert str(waiter_errors[0]) == "owner close operation timed out"


def test_concurrent_close_waiter_does_not_return_before_success() -> None:
    runtime = EmployeeDepartmentRuntime()
    entered = threading.Event()
    release = threading.Event()

    def close_resource() -> None:
        entered.set()
        assert release.wait(2.0)

    resource = _CloseResource(close_resource)
    runtime._channels = resource  # type: ignore[assignment]  # noqa: SLF001

    owner, owner_finished, owner_errors = _start_close(runtime)
    assert entered.wait(1.0)
    waiter, waiter_finished, waiter_errors = _start_close(runtime)
    try:
        assert waiter_finished.wait(0.05) is False
    finally:
        release.set()
        owner.join(timeout=1.0)
        waiter.join(timeout=1.0)

    assert owner_finished.is_set()
    assert waiter_finished.is_set()
    assert owner_errors == []
    assert waiter_errors == []
    assert resource.close_calls == 1


def test_close_waiter_timeout_does_not_cancel_owner_attempt() -> None:
    runtime = EmployeeDepartmentRuntime()
    entered = threading.Event()
    release = threading.Event()

    def close_resource() -> None:
        entered.set()
        assert release.wait(2.0)

    resource = _CloseResource(close_resource)
    runtime._channels = resource  # type: ignore[assignment]  # noqa: SLF001

    owner, owner_finished, owner_errors = _start_close(runtime)
    assert entered.wait(1.0)
    try:
        with pytest.raises(TimeoutError) as raised:
            runtime.close(timeout_seconds=0.01)
        assert type(raised.value).__name__ == "EmployeeRuntimeCloseWaitTimeout"
        assert owner_finished.is_set() is False
    finally:
        release.set()
        owner.join(timeout=1.0)

    assert owner_finished.is_set()
    assert owner_errors == []
    assert resource.close_calls == 1
    runtime.close(timeout_seconds=0.0)


@pytest.mark.parametrize("timeout_seconds", (-1.0, float("nan"), True, "1"))
def test_close_rejects_invalid_wait_timeout(timeout_seconds: object) -> None:
    runtime = EmployeeDepartmentRuntime()

    with pytest.raises(ValueError, match="close wait timeout"):
        runtime.close(timeout_seconds=timeout_seconds)  # type: ignore[arg-type]
