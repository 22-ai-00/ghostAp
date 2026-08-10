"""Crash/restart coverage for the employee provisioning Saga."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning.hire_port import EmployeeHireRequest
from src.autonomous.provisioning.hire_service import ProductionEmployeeHireService
from src.autonomous.provisioning.hire_state import (
    HireProjection,
    HireProjectionError,
)
from src.autonomous.workforce.credential_vault import CredentialKeyring, CredentialVault

HMAC_KEY = b"hire-saga-recovery-hmac-key-32b!"
APP_SECRET = "durable-secret-only-in-vault"


def _request(message_id: str = "om_recover") -> EmployeeHireRequest:
    return EmployeeHireRequest(
        employee_name="Atlas",
        tool="codex",
        model="gpt-5.6-sol",
        effort="high",
        chat_id="oc_admin_dm",
        message_id=message_id,
        requester_principal_id="ou_admin",
        tenant_key="tenant-a",
        profile="standard",
        role="software engineer",
        persona="careful reviewer",
    )


def _writer(tmp_path: Path, epoch: int) -> JournalWriter:
    return JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=epoch,
    )


def _vault(tmp_path: Path) -> CredentialVault:
    return CredentialVault(
        tmp_path / "vault",
        CredentialKeyring(keys={"k1": b"v" * 32}, active_key_id="k1"),
    )


def _service(writer, vault=None, registrar=None):
    return ProductionEmployeeHireService(
        writer,
        ProjectionState(),
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
        credential_vault=vault,
        registrar=registrar,
    )






@pytest.mark.parametrize(
    "events",
    [
        (
            JournalEvent(
                event_type="employee.created",
                aggregate_id="agt_1",
                payload={
                    "agent_id": "agt_1",
                    "tenant_key": "tenant-a",
                    "owner_principal_id": "ou_admin",
                    "name": "Atlas",
                    "tool": "codex",
                    "model": "model",
                    "effort": "high",
                    "profile": "standard",
                    "worker_type": "visible",
                    "state": "provisioning_app",
                    "hire_schema_version": 1,
                    "hire_intent_id": "hire_1",
                    "hire_message_id": "om_1",
                    "hire_chat_id": "oc_1",
                    "planned_bot_principal_id": "bot_1",
                    "provisioning_attempt_id": "attempt_1",
                },
            ),
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id="agt_1",
                payload={"state": "active"},
            ),
        ),
        (
            JournalEvent(
                event_type="employee.created",
                aggregate_id="agt_1",
                payload={
                    "agent_id": "agt_1",
                    "tenant_key": "tenant-a",
                    "owner_principal_id": "ou_admin",
                    "name": "Atlas",
                    "tool": "codex",
                    "model": "model",
                    "effort": "high",
                    "profile": "standard",
                    "worker_type": "visible",
                    "state": "provisioning_app",
                    "hire_schema_version": 1,
                    "hire_intent_id": "hire_1",
                    "hire_message_id": "om_1",
                    "hire_chat_id": "oc_1",
                    "planned_bot_principal_id": "bot_1",
                    "provisioning_attempt_id": "attempt_1",
                },
            ),
            JournalEvent(
                event_type="hire.effect.prepared",
                aggregate_id="hire_1",
                payload={
                    "effect_id": "register-app",
                    "effect_type": "app_registration",
                    "metadata": {"app_secret": "must-reject"},
                },
            ),
        ),
    ],
)
def test_replay_fails_closed_on_illegal_phase_or_effect_metadata(events) -> None:
    class Frame:
        sequence = 1

        def __init__(self, values):
            self.events = values

    with pytest.raises(HireProjectionError):
        HireProjection.rebuild([Frame(events)])  # type: ignore[arg-type]
