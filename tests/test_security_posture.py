"""Typed, fail-closed security-posture contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from src.config import Settings
from src.config.security_posture import (
    IngressAccessMode,
    SecurityFinding,
    SecuritySeverity,
    ShellAccessMode,
    evaluate_security_posture,
)


@pytest.mark.parametrize(
    (
        "mode",
        "admins",
        "users",
        "chats",
        "ack",
        "isolation_ready",
        "valid",
        "expected_code",
    ),
    [
        ("disabled", "", "", "", False, False, True, None),
        ("admin_dm", "", "", "", False, False, False, "shell_admin_missing"),
        ("admin_dm", "ou_admin", "", "", False, False, True, None),
        ("allowlisted", "", "", "", False, False, False, "shell_allowlist_missing"),
        ("allowlisted", "", "ou_user", "", False, False, False, "shell_allowlist_missing"),
        ("allowlisted", "", "", "oc_chat", False, False, False, "shell_allowlist_missing"),
        ("allowlisted", "", "ou_user", "oc_chat", False, False, True, None),
        ("isolated", "", "", "", False, False, False, "shell_isolation_unavailable"),
        ("isolated", "", "", "", False, True, True, None),
        (
            "trusted_local",
            "",
            "",
            "",
            False,
            False,
            False,
            "shell_trusted_local_unacknowledged",
        ),
        ("trusted_local", "", "", "", True, False, True, None),
    ],
)
def test_shell_access_posture_is_fail_closed(
    mode: str,
    admins: str,
    users: str,
    chats: str,
    ack: bool,
    isolation_ready: bool,
    valid: bool,
    expected_code: str | None,
) -> None:
    settings = Settings(
        _env_file=None,
        shell_access_mode=mode,
        shell_trusted_local_ack=ack,
        admin_user_ids=admins,
        allowed_user_ids=users,
        allowed_chat_ids=chats,
        admin_bootstrap_scope="p2p_only",
    )

    posture = evaluate_security_posture(
        settings,
        isolation_ready=isolation_ready,
    )

    assert posture.shell_mode is ShellAccessMode(mode)
    assert posture.is_valid is valid
    codes = {finding.code for finding in posture.findings}
    if expected_code is not None:
        assert expected_code in codes


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("shadow", "ingress_shadow_not_enforcing"),
        ("legacy_allow_all", "ingress_legacy_allow_all"),
    ],
)
def test_non_enforcing_ingress_modes_are_visibly_reported(
    mode: str,
    expected_code: str,
) -> None:
    posture = evaluate_security_posture(
        Settings(
            _env_file=None,
            ingress_access_mode=mode,
            admin_bootstrap_scope="p2p_only",
        )
    )

    assert expected_code in {finding.code for finding in posture.findings}
    assert posture.ingress_mode is IngressAccessMode(mode)


def test_secure_defaults_are_enforced_and_shell_disabled() -> None:
    settings = Settings(_env_file=None)

    assert settings.ingress_access_mode == "enforced"
    assert settings.admin_bootstrap_scope == "p2p_only"
    assert settings.shell_access_mode == "disabled"
    assert settings.shell_trusted_local_ack is False
    assert "employee_department_enabled" not in Settings.model_fields
    assert settings.autonomous_visible_employee_limit == 8
    assert settings.employee_group_context_retention_days == 30

    posture = evaluate_security_posture(settings)
    assert posture.employee_department_enabled is True
    assert posture.records_group_content is True
    assert posture.is_valid
    assert posture.findings == ()


@pytest.mark.parametrize(
    ("visible_employee_limit", "expected_enabled"),
    [(0, False), (1, True), (8, True)],
)
def test_employee_department_posture_tracks_real_composition_gate(
    visible_employee_limit: int,
    expected_enabled: bool,
) -> None:
    posture = evaluate_security_posture(
        Settings(
            _env_file=None,
            autonomous_visible_employee_limit=visible_employee_limit,
        )
    )

    assert posture.employee_department_enabled is expected_enabled
    assert posture.records_group_content is expected_enabled


def test_any_chat_bootstrap_is_distinguishable_from_secure_posture() -> None:
    posture = evaluate_security_posture(
        Settings(_env_file=None, admin_bootstrap_scope="any_chat")
    )

    finding = next(
        item
        for item in posture.findings
        if item.code == "admin_bootstrap_any_chat"
    )
    assert finding.severity is SecuritySeverity.WARNING


def test_employee_group_retention_missing_is_blocking() -> None:
    settings = Settings.model_construct(
        ingress_access_mode="enforced",
        admin_bootstrap_scope="p2p_only",
        shell_access_mode="disabled",
        shell_trusted_local_ack=False,
        autonomous_visible_employee_limit=8,
        employee_group_context_retention_days=0,
        admin_user_ids=frozenset(),
        allowed_user_ids=frozenset(),
        allowed_chat_ids=frozenset(),
    )

    posture = evaluate_security_posture(settings)

    assert not posture.is_valid
    assert "employee_group_retention_missing" in {
        finding.code for finding in posture.findings
    }


def test_security_findings_are_frozen_values() -> None:
    finding = SecurityFinding(
        code="example",
        severity=SecuritySeverity.INFO,
        message="example",
    )

    with pytest.raises(FrozenInstanceError):
        finding.code = "changed"  # type: ignore[misc]


def test_validate_prints_posture_and_exits_nonzero_for_blocker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.main import main

    settings = Settings(
        _env_file=None,
        app_id="cli_test",
        app_secret="cli_test_secret",
        shell_access_mode="admin_dm",
        admin_bootstrap_scope="p2p_only",
        admin_user_ids="",
    )
    with patch("src.main.get_settings", return_value=settings):
        with pytest.raises(SystemExit) as exc_info:
            main(["--validate"])

    assert exc_info.value.code == 1
    output = capsys.readouterr()
    assert "[安全姿态]" in output.out
    assert "shell_admin_missing" in output.out
    assert "安全姿态存在阻断项" in output.err


def test_validate_secure_defaults_print_posture_and_exit_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.main import main

    settings = Settings(
        _env_file=None,
        app_id="cli_test",
        app_secret="cli_test_secret",
    )
    with patch("src.main.get_settings", return_value=settings):
        with pytest.raises(SystemExit) as exc_info:
            main(["--validate"])

    assert exc_info.value.code == 0
    output = capsys.readouterr()
    assert "[安全姿态]" in output.out
    assert "INGRESS_ACCESS_MODE = enforced" in output.out
    assert "SHELL_ACCESS_MODE   = disabled" in output.out
    assert "配置校验通过" in output.out
