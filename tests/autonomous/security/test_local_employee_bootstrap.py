from __future__ import annotations

import errno
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.autonomous.provisioning import local_bootstrap as local_bootstrap_module
from src.autonomous.provisioning.local_bootstrap import (
    LocalEmployeeBootstrapError,
    load_or_create_local_employee_material,
    resolve_employee_runtime_material,
)
from src.config.settings import Settings


def test_bootstrap_creates_private_stable_independent_material(tmp_path: Path) -> None:
    first = load_or_create_local_employee_material(tmp_path)
    second = load_or_create_local_employee_material(tmp_path)

    assert first == second
    assert first.journal_hmac_key not in {
        next(iter(first.credential_keyring.keys.values())),
        next(iter(first.data_keyring.keys.values())),
    }
    secret_path = tmp_path / "employee-runtime-secrets.json"
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    assert secret_path.stat().st_uid == os.geteuid()


def test_bootstrap_concurrent_callers_converge(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        materials = tuple(
            executor.map(
                lambda _: load_or_create_local_employee_material(tmp_path),
                range(24),
            )
        )

    assert all(material == materials[0] for material in materials)


def test_bootstrap_retries_transient_lock_creation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open
    lock_attempts = 0

    def flaky_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal lock_attempts
        if path == local_bootstrap_module._LOCK_FILENAME:
            lock_attempts += 1
            if lock_attempts == 1:
                raise FileNotFoundError(errno.ENOENT, "transient openat race", path)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(local_bootstrap_module.os, "open", flaky_open)

    material = load_or_create_local_employee_material(tmp_path)

    assert material.journal_hmac_key
    assert lock_attempts == 2


@pytest.mark.parametrize(
    "payload",
    (
        "{",
        '{"version":1,"version":1}',
        json.dumps({"version": 1}),
    ),
)
def test_bootstrap_rejects_malformed_existing_envelope(
    tmp_path: Path,
    payload: str,
) -> None:
    tmp_path.chmod(0o700)
    secret_path = tmp_path / "employee-runtime-secrets.json"
    secret_path.write_text(payload, encoding="utf-8")
    secret_path.chmod(0o600)

    with pytest.raises(LocalEmployeeBootstrapError) as raised:
        load_or_create_local_employee_material(tmp_path)

    assert str(raised.value) == "LocalEmployeeBootstrapError"


def test_bootstrap_rejects_symlink_and_overly_broad_mode(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    secret_path = tmp_path / "employee-runtime-secrets.json"
    secret_path.symlink_to(target)

    with pytest.raises(LocalEmployeeBootstrapError):
        load_or_create_local_employee_material(tmp_path)

    secret_path.unlink()
    material = load_or_create_local_employee_material(tmp_path)
    assert material.journal_hmac_key
    secret_path.chmod(0o640)
    with pytest.raises(LocalEmployeeBootstrapError):
        load_or_create_local_employee_material(tmp_path)


def test_complete_explicit_material_takes_precedence(tmp_path: Path) -> None:
    generated = load_or_create_local_employee_material(tmp_path / "generated")
    credential_key = next(iter(generated.credential_keyring.keys.values()))
    data_key = next(iter(generated.data_keyring.keys.values()))
    settings = Settings.model_construct(
        autonomous_state_dir=str(tmp_path / "unused"),
        autonomous_journal_hmac_key=SecretStr(_b64(generated.journal_hmac_key)),
        autonomous_credential_keys=SecretStr(
            json.dumps({"version": 1, "keys": {"credential-v1": _b64(credential_key)}})
        ),
        autonomous_credential_active_key_id="credential-v1",
        autonomous_data_keys=SecretStr(
            json.dumps({"version": 1, "keys": {"data-v1": _b64(data_key)}})
        ),
        autonomous_data_active_key_id="data-v1",
    )

    resolved = resolve_employee_runtime_material(settings)

    assert resolved.journal_hmac_key == generated.journal_hmac_key
    assert resolved.credential_keyring.keys["credential-v1"] == credential_key
    assert resolved.data_keyring.keys["data-v1"] == data_key
    assert not (tmp_path / "unused").exists()


def test_partial_explicit_material_is_rejected(tmp_path: Path) -> None:
    settings = Settings.model_construct(
        autonomous_state_dir=str(tmp_path),
        autonomous_journal_hmac_key=SecretStr(_b64(os.urandom(32))),
        autonomous_credential_keys=SecretStr(""),
        autonomous_credential_active_key_id="",
        autonomous_data_keys=SecretStr(""),
        autonomous_data_active_key_id="",
    )

    with pytest.raises(LocalEmployeeBootstrapError) as raised:
        resolve_employee_runtime_material(settings)

    assert str(raised.value) == "LocalEmployeeBootstrapError"
    assert not (tmp_path / "employee-runtime-secrets.json").exists()


def _b64(value: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(value).decode("ascii")
