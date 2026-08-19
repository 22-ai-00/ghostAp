"""Contracts for the opt-in, per-task ACP trusted-personal lease."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

from acp.schema import PermissionOption

from src.acp.client import GhostAPClient
from src.acp.personal_trust import (
    TrustedPersonalPermissionLease,
    trusted_personal_permissions_requested,
)
from src.acp.session import ACPSession
from src.config import Settings
from src.config.security_posture import SecuritySeverity, evaluate_security_posture


def _trusted_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "acp_trusted_personal_mode": True,
        "acp_trusted_personal_ack": True,
        "admin_user_ids": frozenset({"ou_owner"}),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_trusted_personal_mode_is_secure_by_default() -> None:
    settings = Settings(_env_file=None)

    assert settings.acp_trusted_personal_mode is False
    assert settings.acp_trusted_personal_ack is False


def test_trusted_personal_mode_requires_ack_and_admin() -> None:
    missing_ack = evaluate_security_posture(
        Settings(
            _env_file=None,
            acp_trusted_personal_mode=True,
            acp_trusted_personal_ack=False,
            admin_user_ids="ou_owner",
        )
    )
    missing_admin = evaluate_security_posture(
        Settings(
            _env_file=None,
            acp_trusted_personal_mode=True,
            acp_trusted_personal_ack=True,
            admin_user_ids="",
        )
    )
    active = evaluate_security_posture(
        Settings(
            _env_file=None,
            acp_trusted_personal_mode=True,
            acp_trusted_personal_ack=True,
            admin_user_ids="ou_owner",
        )
    )

    assert not missing_ack.is_valid
    assert "acp_trusted_personal_unacknowledged" in {
        finding.code for finding in missing_ack.findings
    }
    assert not missing_admin.is_valid
    assert "acp_trusted_personal_admin_missing" in {
        finding.code for finding in missing_admin.findings
    }
    active_finding = next(
        finding
        for finding in active.findings
        if finding.code == "acp_trusted_personal_active"
    )
    assert active_finding.severity is SecuritySeverity.WARNING
    assert active.is_valid


def test_trusted_personal_scope_requires_admin_and_current_project() -> None:
    project = SimpleNamespace(project_id="project-1", root_path="/repo")

    assert trusted_personal_permissions_requested(
        _trusted_settings(),
        project=project,
        sender_id="ou_owner",
    )
    assert not trusted_personal_permissions_requested(
        _trusted_settings(),
        project=None,
        sender_id="ou_owner",
    )
    assert not trusted_personal_permissions_requested(
        _trusted_settings(),
        project=project,
        sender_id="ou_other",
    )
    assert not trusted_personal_permissions_requested(
        _trusted_settings(acp_trusted_personal_ack=False),
        project=project,
        sender_id="ou_owner",
    )


def test_trusted_client_approves_git_push_despite_generic_policy_gates() -> None:
    client = GhostAPClient(on_event=lambda _event: None, auto_approve=False)
    client.set_tool_filter(lambda _tool, _args: False)
    client.set_trusted_personal_permissions(True)
    option = PermissionOption(
        optionId="allow-session",
        name="Allow for session",
        kind="allow_always",
    )
    tool_call = SimpleNamespace(
        kind="execute",
        raw_input={"command": ["git", "push", "origin", "dev"]},
    )

    allowed = asyncio.run(
        client.request_permission(
            session_id="session-1",
            tool_call=tool_call,
            options=[option],
        )
    )
    client.set_trusted_personal_permissions(False)
    denied = asyncio.run(
        client.request_permission(
            session_id="session-1",
            tool_call=tool_call,
            options=[option],
        )
    )

    assert allowed.outcome.outcome == "selected"
    assert allowed.outcome.option_id == "allow-session"
    assert denied.outcome.outcome == "cancelled"


def test_official_codex_trusted_lease_switches_full_access_then_restores() -> None:
    session = ACPSession(
        agent_cmd="npx",
        agent_args=["-y", "@agentclientprotocol/codex-acp@1.2.0"],
        cwd="/repo",
        env={"INITIAL_AGENT_MODE": "read-only"},
    )
    session._conn = object()  # noqa: SLF001
    session._session_id = "codex-session"  # noqa: SLF001
    session._client = MagicMock()  # noqa: SLF001
    session.set_config_option = AsyncMock(return_value=True)  # type: ignore[method-assign]

    assert asyncio.run(session.set_trusted_personal_permissions(True))
    assert asyncio.run(session.set_trusted_personal_permissions(False))

    assert session.set_config_option.await_args_list == [
        call("mode", "agent-full-access"),
        call("mode", "read-only"),
    ]
    assert session._client.set_trusted_personal_permissions.call_args_list == [  # noqa: SLF001
        call(True),
        call(False),
    ]


def test_trusted_lease_applies_to_replacements_and_always_revokes() -> None:
    first = MagicMock()
    first.set_trusted_personal_permissions.return_value = True
    replacement = MagicMock()
    replacement.set_trusted_personal_permissions.return_value = True
    lease = TrustedPersonalPermissionLease(enabled=True)

    lease.acquire(first)
    lease.acquire(replacement)
    lease.acquire(replacement)
    failures = lease.release_all()

    assert failures == ()
    first.set_trusted_personal_permissions.assert_has_calls([call(True), call(False)])
    replacement.set_trusted_personal_permissions.assert_has_calls(
        [call(True), call(False)]
    )


def test_trusted_lease_marks_session_dead_when_revocation_fails() -> None:
    session = MagicMock()
    session.set_trusted_personal_permissions.side_effect = [True, False]
    lease = TrustedPersonalPermissionLease(enabled=True)

    lease.acquire(session)
    failures = lease.release_all()

    assert failures == (session,)
    assert session._force_dead is True  # noqa: SLF001
