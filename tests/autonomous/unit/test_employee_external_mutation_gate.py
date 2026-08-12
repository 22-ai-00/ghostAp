from __future__ import annotations

import threading
import time

import pytest

from src.autonomous.provisioning.external_mutation_gate import (
    EmployeeExternalMutationGate,
    ExternalMutationFenced,
    ExternalMutationKind,
)


@pytest.mark.parametrize("kind", tuple(ExternalMutationKind))
def test_retirement_waits_for_each_hire_side_effect_lease(
    kind: ExternalMutationKind,
) -> None:
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire("tenant_1", "agt_1", kind)
    started = threading.Event()
    finished = threading.Event()
    result: list[bool] = []

    def retire() -> None:
        started.set()
        result.append(
            gate.begin_retirement(
                "tenant_1",
                "agt_1",
                timeout_seconds=1.0,
            )
        )
        finished.set()

    thread = threading.Thread(target=retire)
    thread.start()
    assert started.wait(1.0)
    assert not finished.wait(0.05)

    # A lease that linearized before the retirement fence remains valid long
    # enough to record its returned external evidence. Fire cannot admit the
    # RETIRING fact until that commit finishes and the lease is released.
    lease.require_active()
    lease.release()

    assert finished.wait(1.0)
    thread.join(timeout=1.0)
    assert result == [True]
    assert gate.is_fenced("tenant_1", "agt_1") is True


@pytest.mark.parametrize("kind", tuple(ExternalMutationKind))
def test_retirement_fence_prevents_every_new_hire_side_effect(
    kind: ExternalMutationKind,
) -> None:
    gate = EmployeeExternalMutationGate()
    assert (
        gate.begin_retirement(
            "tenant_1",
            "agt_1",
            timeout_seconds=0.01,
        )
        is True
    )

    with pytest.raises(ExternalMutationFenced, match="retiring"):
        gate.acquire("tenant_1", "agt_1", kind)


def test_retirement_timeout_is_fail_closed_and_does_not_deadlock() -> None:
    gate = EmployeeExternalMutationGate()
    lease = gate.acquire(
        "tenant_1",
        "agt_1",
        ExternalMutationKind.APP_REGISTRATION,
    )

    started = time.monotonic()
    assert (
        gate.begin_retirement(
            "tenant_1",
            "agt_1",
            timeout_seconds=0.02,
        )
        is False
    )
    assert time.monotonic() - started < 0.5
    assert gate.is_fenced("tenant_1", "agt_1") is True
    with pytest.raises(ExternalMutationFenced):
        gate.acquire(
            "tenant_1",
            "agt_1",
            ExternalMutationKind.CHANNEL_START,
        )

    lease.release()
    assert (
        gate.begin_retirement(
            "tenant_1",
            "agt_1",
            timeout_seconds=0.1,
        )
        is True
    )


def test_restart_restores_a_permanent_retirement_fence() -> None:
    gate = EmployeeExternalMutationGate()
    gate.restore_retirement_fence("tenant_1", "agt_archived")

    assert gate.is_fenced("tenant_1", "agt_archived") is True
    with pytest.raises(ExternalMutationFenced):
        gate.acquire(
            "tenant_1",
            "agt_archived",
            ExternalMutationKind.MEMBERSHIP_ADD,
        )


def test_same_agent_id_in_another_tenant_is_not_fenced() -> None:
    gate = EmployeeExternalMutationGate()
    gate.restore_retirement_fence("tenant_1", "agt_shared")

    lease = gate.acquire(
        "tenant_2",
        "agt_shared",
        ExternalMutationKind.APP_REGISTRATION,
    )

    lease.release()
