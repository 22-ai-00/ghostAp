import base64
import json
import tomllib
from pathlib import Path

import pytest

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


def test_env_example_hides_internal_context_tuning_but_keeps_safety_bounds() -> None:
    active_keys = {
        line.split("=", 1)[0]
        for line in Path(".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    internal_fields = {
        "autonomous_thread_context_max_messages",
        "autonomous_thread_context_max_chars",
        "autonomous_group_context_max_messages",
        "autonomous_thread_context_page_size",
        "autonomous_group_context_page_size",
        "autonomous_context_fetch_timeout_seconds",
        "autonomous_fire_grace_seconds",
        "autonomous_context_max_pages",
    }

    assert {
        name.upper() for name in internal_fields
    }.isdisjoint(active_keys)
    assert "AUTONOMOUS_CONTEXT_MAX_TOKENS" in active_keys
    assert internal_fields <= Settings.model_fields.keys()
