import base64
import json
import tomllib
from pathlib import Path

import pytest

from src.autonomous.config import (
    AutonomousDeploymentMode,
    EffectiveAutonomy,
    derive_effective_autonomy,
)
from src.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_locked_lark_dependencies() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert "lark-oapi==1.7.1" in project["project"]["dependencies"]
    assert "lark-channel-sdk==1.1.0" in project["project"]["dependencies"]
    assert "cryptography==49.0.0" in project["project"]["dependencies"]


def test_employee_credential_settings_default_fail_closed_and_redact(settings: Settings) -> None:
    empty = settings
    assert empty.autonomous_employee_storage_base == "~/.ghostap/slock"
    assert empty.autonomous_credential_dir == "~/.ghostap/slock/credentials"
    assert empty.autonomous_credential_keys.get_secret_value() == ""
    assert empty.autonomous_credential_active_key_id == ""

    encoded = base64.urlsafe_b64encode(bytes([7]) * 32).decode()
    keyring_json = json.dumps({"version": 1, "keys": {"k1": encoded}})
    configured = Settings(
        _env_file=None,
        autonomous_credential_keys=keyring_json,
        autonomous_credential_active_key_id="k1",
    )
    assert keyring_json not in repr(configured)


def test_only_two_employee_data_paths_remain_explicit_continuity_abi() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")

    continuity_abi_paths = {
        line
        for line in env_example.splitlines()
        if line.startswith(
            (
                "AUTONOMOUS_EMPLOYEE_STORAGE_BASE=",
                "AUTONOMOUS_CREDENTIAL_DIR=",
            )
        )
    }
    assert continuity_abi_paths == {
        "AUTONOMOUS_EMPLOYEE_STORAGE_BASE=~/.ghostap/slock",
        "AUTONOMOUS_CREDENTIAL_DIR=~/.ghostap/slock/credentials",
    }
    assert "SLOCK_" not in env_example
    assert "slock_auto_activate_default_policy" not in Settings.model_fields
    assert "slock_default_wake_policy" not in Settings.model_fields
    assert "slock_default_roles" not in Settings.model_fields
    assert "autonomous_compatibility_mode" not in Settings.model_fields
    assert "AUTONOMOUS_COMPATIBILITY_MODE=" not in env_example


def test_employee_data_settings_default_fail_closed_and_redact(settings: Settings) -> None:
    assert settings.autonomous_data_keys.get_secret_value() == ""
    assert settings.autonomous_data_active_key_id == ""
    assert settings.autonomous_data_blob_dir == "~/.ghostap/autonomy/data-blobs"
    assert settings.autonomous_history_timezone == "UTC"
    assert settings.autonomous_history_max_range_days == 31
    assert settings.autonomous_history_page_size == 50

    encoded = base64.urlsafe_b64encode(bytes([9]) * 32).decode()
    keyring_json = json.dumps({"version": 1, "keys": {"data-v1": encoded}})
    configured = Settings(
        _env_file=None,
        autonomous_data_keys=keyring_json,
        autonomous_data_active_key_id="data-v1",
    )
    assert keyring_json not in repr(configured)


def test_employee_data_settings_validate_timezone_and_query_bounds() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, autonomous_history_timezone="Mars/Olympus")
    with pytest.raises(ValueError):
        Settings(_env_file=None, autonomous_history_max_range_days=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, autonomous_history_page_size=201)


def test_employee_thread_context_settings_defaults(settings: Settings) -> None:
    assert settings.autonomous_thread_context_max_messages == 200
    assert settings.autonomous_thread_context_max_chars == 400_000
    assert settings.autonomous_group_context_max_messages == 50
    assert settings.autonomous_context_max_tokens == 128_000
    assert settings.autonomous_thread_context_page_size == 50
    assert settings.autonomous_group_context_page_size == 20
    assert settings.autonomous_context_fetch_timeout_seconds == 30.0
    assert settings.autonomous_fire_grace_seconds == 30.0
    assert settings.autonomous_context_max_pages == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("autonomous_thread_context_max_messages", 0),
        ("autonomous_thread_context_max_chars", 0),
        ("autonomous_group_context_max_messages", 0),
        ("autonomous_context_max_tokens", 0),
        ("autonomous_thread_context_page_size", 0),
        ("autonomous_thread_context_page_size", 51),
        ("autonomous_group_context_page_size", 0),
        ("autonomous_group_context_page_size", 51),
        ("autonomous_context_fetch_timeout_seconds", 0),
        ("autonomous_context_fetch_timeout_seconds", float("inf")),
        ("autonomous_fire_grace_seconds", 0),
        ("autonomous_fire_grace_seconds", float("inf")),
        ("autonomous_context_max_pages", 0),
        ("autonomous_thread_context_max_messages", True),
        ("autonomous_context_fetch_timeout_seconds", True),
        ("autonomous_fire_grace_seconds", True),
    ],
)
def test_employee_thread_context_settings_reject_invalid_bounds(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


def test_env_example_documents_employee_thread_context_settings() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")
    for name in (
        "AUTONOMOUS_THREAD_CONTEXT_MAX_MESSAGES",
        "AUTONOMOUS_THREAD_CONTEXT_MAX_CHARS",
        "AUTONOMOUS_GROUP_CONTEXT_MAX_MESSAGES",
        "AUTONOMOUS_CONTEXT_MAX_TOKENS",
        "AUTONOMOUS_THREAD_CONTEXT_PAGE_SIZE",
        "AUTONOMOUS_GROUP_CONTEXT_PAGE_SIZE",
        "AUTONOMOUS_CONTEXT_FETCH_TIMEOUT_SECONDS",
        "AUTONOMOUS_FIRE_GRACE_SECONDS",
        "AUTONOMOUS_CONTEXT_MAX_PAGES",
    ):
        assert f"{name}=" in env_example


def test_autonomous_settings_are_fail_closed_by_default(settings: Settings) -> None:
    assert settings.autonomous_deployment_mode == AutonomousDeploymentMode.OFF
    assert settings.autonomous_memory_enabled is False
    assert settings.autonomous_vc_enabled is False
    assert settings.autonomous_write_enabled is False
    assert settings.autonomous_state_dir == "~/.ghostap/autonomy"
    assert settings.autonomous_journal_dir == "~/.ghostap/autonomy/journal"
    assert settings.autonomous_snapshot_dir == "~/.ghostap/autonomy/snapshots"
    assert settings.autonomous_manager_acl == ""
    assert settings.autonomous_anchor_provider == ""
    assert settings.autonomous_sandbox_required is True
    assert settings.autonomous_goal_queue_limit == 1000
    assert settings.autonomous_run_queue_limit == 100


def test_deployment_mode_values_are_frozen() -> None:
    assert {mode.value for mode in AutonomousDeploymentMode} == {
        "off",
        "shadow_read",
        "manager_only",
    }
    assert {mode.value for mode in EffectiveAutonomy} == {
        "off",
        "shadow_read",
        "assist",
        "supervised",
        "bounded_autonomous",
    }


def test_write_modes_fail_closed_without_anchor_and_sandbox(settings: Settings) -> None:
    settings.autonomous_deployment_mode = "manager_only"

    status = derive_effective_autonomy(settings, {})

    assert status.mode is EffectiveAutonomy.ASSIST
    assert set(status.blockers) == {
        "journal",
        "anchor",
        "worker_sandbox",
        "oracle_sandbox",
        "brokers",
        "p0_gates",
        "write_enabled",
        "manager_acl",
    }


def test_manager_only_reaches_supervised_with_required_attestations(settings: Settings) -> None:
    settings.autonomous_deployment_mode = "manager_only"
    settings.autonomous_write_enabled = True
    settings.autonomous_manager_acl = "ou_manager"
    settings.autonomous_anchor_provider = "verified-cas"
    attestations = {
        "journal": True,
        "anchor": True,
        "worker_sandbox": True,
        "oracle_sandbox": True,
        "brokers": True,
        "p0_gates": True,
    }

    status = derive_effective_autonomy(settings, attestations)

    assert status.mode is EffectiveAutonomy.SUPERVISED
    assert status.blockers == ()
    assert status.attestations == attestations


@pytest.mark.parametrize(
    ("deployment_mode", "expected"),
    [
        ("off", EffectiveAutonomy.OFF),
        ("shadow_read", EffectiveAutonomy.SHADOW_READ),
    ],
)
def test_non_write_deployment_modes_do_not_escalate(
    settings: Settings,
    deployment_mode: str,
    expected: EffectiveAutonomy,
) -> None:
    settings.autonomous_deployment_mode = deployment_mode

    status = derive_effective_autonomy(
        settings,
        {
            "journal": True,
            "anchor": True,
            "worker_sandbox": True,
            "oracle_sandbox": True,
            "brokers": True,
            "p0_gates": True,
            "standing_order": True,
        },
    )

    assert status.mode is expected
    assert status.blockers == ()


def test_bounded_mode_requires_a_standing_order(settings: Settings) -> None:
    settings.autonomous_deployment_mode = "manager_only"
    settings.autonomous_write_enabled = True
    settings.autonomous_manager_acl = "ou_manager"
    settings.autonomous_anchor_provider = "verified-cas"
    base_attestations = {
        "journal": True,
        "anchor": True,
        "worker_sandbox": True,
        "oracle_sandbox": True,
        "brokers": True,
        "p0_gates": True,
        "standing_order": False,
    }

    supervised = derive_effective_autonomy(settings, base_attestations)
    bounded = derive_effective_autonomy(
        settings,
        {**base_attestations, "standing_order": True},
    )

    assert supervised.mode is EffectiveAutonomy.SUPERVISED
    assert bounded.mode is EffectiveAutonomy.BOUNDED_AUTONOMOUS


def test_attestations_must_be_literal_booleans(settings: Settings) -> None:
    settings.autonomous_deployment_mode = "manager_only"
    settings.autonomous_write_enabled = True
    settings.autonomous_manager_acl = "ou_manager"
    settings.autonomous_anchor_provider = "verified-cas"

    status = derive_effective_autonomy(
        settings,
        {
            "journal": "false",
            "anchor": 1,
            "worker_sandbox": True,
            "oracle_sandbox": True,
            "brokers": True,
            "p0_gates": True,
            "standing_order": "false",
        },
    )

    assert status.mode is EffectiveAutonomy.ASSIST
    assert {"journal", "anchor"} <= set(status.blockers)
    assert status.attestations["journal"] is False
    assert status.attestations["anchor"] is False
    assert status.attestations["standing_order"] is False


def test_any_p0_forces_offline_assist(settings: Settings) -> None:
    settings.autonomous_deployment_mode = "manager_only"
    settings.autonomous_write_enabled = True
    settings.autonomous_manager_acl = "ou_manager"
    settings.autonomous_anchor_provider = "verified-cas"

    status = derive_effective_autonomy(
        settings,
        {
            "journal": True,
            "anchor": True,
            "worker_sandbox": True,
            "oracle_sandbox": True,
            "brokers": True,
            "p0_gates": False,
            "standing_order": True,
        },
    )

    assert status.mode is EffectiveAutonomy.ASSIST
    assert status.blockers == ("p0_gates",)
