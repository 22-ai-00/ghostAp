"""Tests for /hire provisioning saga."""

from __future__ import annotations

import time

import pytest

from src.autonomous.provisioning import (
    HireIntent,
    HireSaga,
    SagaPhase,
    SagaStateError,
)


class _FakeAppCreation:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls: list[dict] = []

    def generate_creation_link(self, *, app_name, description, scopes, event_subscriptions):
        self.calls.append({"app_name": app_name})
        if self._fail:
            raise RuntimeError("SDK error")
        return ("https://open.feishu.cn/create/abc123", time.time() + 600)

    def validate_callback(self, callback_payload):
        return (callback_payload["app_id"], callback_payload["app_secret"])


class _FakeVault:
    def __init__(self) -> None:
        self.stored: list[dict] = []

    def store(self, *, agent_id, app_id, app_secret, hire_intent_id, attempt_id):
        self.stored.append({"agent_id": agent_id, "app_id": app_id})
        return f"cred_ref_{app_id}"


class _FakeJournal:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_hire_event(self, *, intent_id, phase, payload):
        self.events.append({"intent_id": intent_id, "phase": phase, "payload": payload})


def _intent(name: str = "CodeBot") -> HireIntent:
    return HireIntent(
        intent_id=f"hire_{name.lower()}",
        employee_name=name,
        tool="codex",
        model="gpt-test",
        effort="high",
        tenant_key="tenant_1",
        owner_principal_id="principal_owner",
        chat_id="chat_1",
    )


class TestHireSaga:
    def test_full_happy_path(self) -> None:
        app = _FakeAppCreation()
        vault = _FakeVault()
        journal = _FakeJournal()
        saga = HireSaga(app_creation=app, vault=vault, journal=journal)

        state = saga.initiate(_intent())
        assert state.phase == SagaPhase.LINK_GENERATED
        assert state.creation_link.startswith("https://")

        state = saga.on_creation_callback(
            "hire_codebot",
            {"app_id": "cli_abc", "app_secret": "secret_123"},
        )
        assert state.phase == SagaPhase.VAULT_STORED
        assert state.app_id == "cli_abc"
        assert state.credential_ref == "cred_ref_cli_abc"

        state = saga.mark_channel_connected("hire_codebot")
        assert state.phase == SagaPhase.CHANNEL_CONNECTED

        state = saga.mark_slash_registered("hire_codebot")
        assert state.phase == SagaPhase.SLASH_REGISTERED

        state = saga.complete("hire_codebot")
        assert state.phase == SagaPhase.COMPLETED
        assert state.is_terminal

        assert len(journal.events) == 5
        assert vault.stored[0]["app_id"] == "cli_abc"

    def test_sdk_failure_marks_failed(self) -> None:
        saga = HireSaga(
            app_creation=_FakeAppCreation(fail=True),
            vault=_FakeVault(),
            journal=_FakeJournal(),
        )
        state = saga.initiate(_intent())
        assert state.phase == SagaPhase.FAILED
        assert "SDK error" in state.error

    def test_expired_link_fails_on_callback(self) -> None:
        app = _FakeAppCreation()
        saga = HireSaga(app_creation=app, vault=_FakeVault(), journal=_FakeJournal())
        state = saga.initiate(_intent())
        state.link_expires_at = time.time() - 1
        result = saga.on_creation_callback("hire_codebot", {"app_id": "x", "app_secret": "y"})
        assert result.phase == SagaPhase.FAILED
        assert "expired" in result.error

    def test_cancel_in_progress(self) -> None:
        saga = HireSaga(
            app_creation=_FakeAppCreation(),
            vault=_FakeVault(),
            journal=_FakeJournal(),
        )
        saga.initiate(_intent())
        state = saga.cancel("hire_codebot")
        assert state.phase == SagaPhase.CANCELLED
        assert state.is_terminal

    def test_duplicate_initiate_returns_existing(self) -> None:
        saga = HireSaga(
            app_creation=_FakeAppCreation(),
            vault=_FakeVault(),
            journal=_FakeJournal(),
        )
        first = saga.initiate(_intent())
        second = saga.initiate(_intent())
        assert first is second

    def test_wrong_phase_transition_raises(self) -> None:
        saga = HireSaga(
            app_creation=_FakeAppCreation(),
            vault=_FakeVault(),
            journal=_FakeJournal(),
        )
        saga.initiate(_intent())
        with pytest.raises(SagaStateError, match="expected phase"):
            saga.mark_channel_connected("hire_codebot")

    def test_callback_on_unknown_saga_raises(self) -> None:
        saga = HireSaga(
            app_creation=_FakeAppCreation(),
            vault=_FakeVault(),
            journal=_FakeJournal(),
        )
        with pytest.raises(SagaStateError, match="no active saga"):
            saga.on_creation_callback("nonexistent", {})
