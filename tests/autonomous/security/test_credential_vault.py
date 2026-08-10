import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from src.autonomous.workforce.credential_vault import (
    CredentialKeyring,
    CredentialVault,
    CredentialVaultConfigurationError,
    CredentialVaultError,
)


def _key(byte: int) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).decode()


def _settings(payload: str, active_key_id: str = "k1") -> SimpleNamespace:
    return SimpleNamespace(
        autonomous_credential_keys=SecretStr(payload),
        autonomous_credential_active_key_id=active_key_id,
    )


def _vault(tmp_path, *, key_id: str = "k1", byte: int = 1) -> CredentialVault:
    return CredentialVault(
        tmp_path / "credentials",
        CredentialKeyring(keys={key_id: _key(byte)}, active_key_id=key_id),
    )


def _put(vault: CredentialVault, *, secret: str = "secret"):
    return vault.put(
        agent_id="agt_1",
        app_id="cli_1",
        app_secret=secret,
        hire_intent_id="hire_1",
        attempt_id="attempt_1",
    )


def _rewrite_envelope(path: Path, field: str, value) -> None:
    envelope = json.loads(path.read_text())
    envelope[field] = value
    path.write_text(json.dumps(envelope))


def test_keyring_parses_versioned_rotation_set_and_rejects_missing_active() -> None:
    settings = _settings(
        json.dumps({"version": 1, "keys": {"old": _key(1), "new": _key(2)}}),
        active_key_id="new",
    )
    assert CredentialKeyring.from_settings(settings).active_key_id == "new"
    settings.autonomous_credential_active_key_id = "absent"
    with pytest.raises(CredentialVaultConfigurationError):
        CredentialKeyring.from_settings(settings)


def test_keyring_repr_redacts_encoded_and_decoded_key_material() -> None:
    encoded = _key(7)
    decoded_repr = repr(bytes([7]) * 32)

    keyring = CredentialKeyring(keys={"k1": encoded}, active_key_id="k1")

    assert encoded not in repr(keyring)
    assert decoded_repr not in repr(keyring)


@pytest.mark.parametrize(
    "payload,active_key_id",
    [
        ("", ""),
        (json.dumps({"version": 1, "keys": {"k1": _key(1)}, "extra": True}), "k1"),
        (json.dumps({"version": 2, "keys": {"k1": _key(1)}}), "k1"),
        ('{"version":1,"keys":{"k1":"%s","k1":"%s"}}' % (_key(1), _key(2)), "k1"),
        (json.dumps({"version": 1, "keys": {"k1": "not-base64!"}}), "k1"),
        (json.dumps({"version": 1, "keys": {"k1": base64.urlsafe_b64encode(b"short").decode()}}), "k1"),
        (json.dumps({"version": 1, "keys": {}}), "k1"),
    ],
)
def test_keyring_rejects_malformed_or_unsafe_settings_without_disclosure(
    payload: str,
    active_key_id: str,
) -> None:
    with pytest.raises(CredentialVaultConfigurationError) as raised:
        CredentialKeyring.from_settings(_settings(payload, active_key_id))

    if payload:
        assert payload not in str(raised.value)
    assert _key(1) not in str(raised.value)
    assert str(raised.value) == "CredentialVaultConfigurationError"


def test_vault_encrypts_secret_and_enforces_modes(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault, secret="super-secret")

    raw = receipt.path.read_bytes()
    assert b"super-secret" not in raw
    assert os.stat(receipt.path).st_mode & 0o777 == 0o600
    assert os.stat(receipt.path.parent).st_mode & 0o777 == 0o700
    assert vault.resolve(receipt.credential_ref, agent_id="agt_1", app_id="cli_1") == "super-secret"


def test_vault_writes_exact_envelope_and_non_secret_deterministic_ref(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault, secret="first-secret")
    envelope = json.loads(receipt.path.read_text())

    assert set(envelope) == {
        "schema_version",
        "credential_ref",
        "key_id",
        "agent_id",
        "app_id",
        "hire_intent_id",
        "attempt_id",
        "nonce",
        "ciphertext",
        "ciphertext_sha256",
        "created_at",
    }
    expected_ref = "cred_" + hashlib.sha256(b"hire_1|attempt_1").hexdigest()
    assert receipt.credential_ref == expected_ref
    assert envelope["credential_ref"] == expected_ref
    assert envelope["ciphertext_sha256"] == hashlib.sha256(
        base64.urlsafe_b64decode(envelope["ciphertext"])
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        (field, invalid_value)
        for field in (
            "agent_id",
            "app_id",
            "hire_intent_id",
            "attempt_id",
            "app_secret",
        )
        for invalid_value in ("", None, 7)
    ],
)
def test_vault_put_rejects_values_the_reader_would_reject_without_writing(
    tmp_path,
    field: str,
    invalid_value: object,
) -> None:
    vault = _vault(tmp_path)
    values: dict[str, object] = {
        "agent_id": "agt_1",
        "app_id": "cli_1",
        "app_secret": "secret-that-must-not-leak",
        "hire_intent_id": "hire_1",
        "attempt_id": "attempt_1",
    }
    values[field] = invalid_value

    with pytest.raises(CredentialVaultError) as raised:
        vault.put(**values)

    assert str(raised.value) == "CredentialVaultError:invalid-input"
    assert "secret-that-must-not-leak" not in str(raised.value)
    assert list((tmp_path / "credentials").glob("*.json")) == []


def test_vault_put_output_immediately_resolves_and_orphan_scans(tmp_path) -> None:
    vault = _vault(tmp_path)

    receipt = _put(vault, secret="producer-parser-secret")

    assert vault.resolve(
        receipt.credential_ref,
        agent_id=receipt.agent_id,
        app_id=receipt.app_id,
    ) == "producer-parser-secret"
    assert vault.find_orphan_receipts(set()) == [receipt]


def test_vault_rejects_wrong_associated_identity(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault)

    with pytest.raises(CredentialVaultError) as raised:
        vault.resolve(receipt.credential_ref, agent_id="agt_2", app_id="cli_1")

    assert str(raised.value) == f"CredentialVaultError:{receipt.credential_ref}"


def test_vault_rejects_tampered_ciphertext_without_disclosure(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault, secret="do-not-disclose")
    envelope = json.loads(receipt.path.read_text())
    ciphertext = bytearray(base64.urlsafe_b64decode(envelope["ciphertext"]))
    ciphertext[-1] ^= 1
    envelope["ciphertext"] = base64.urlsafe_b64encode(ciphertext).decode()
    receipt.path.write_text(json.dumps(envelope))

    with pytest.raises(CredentialVaultError) as raised:
        vault.resolve(receipt.credential_ref, agent_id="agt_1", app_id="cli_1")

    assert str(raised.value) == f"CredentialVaultError:{receipt.credential_ref}"
    assert "do-not-disclose" not in str(raised.value)
    assert _key(1) not in str(raised.value)


def test_vault_finds_orphan_receipt(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault)

    assert [r.credential_ref for r in vault.find_orphan_receipts(set())] == [
        receipt.credential_ref
    ]
    assert vault.find_orphan_receipts({receipt.credential_ref}) == []


def test_vault_authenticates_key_id_even_when_key_bytes_match(tmp_path) -> None:
    root = tmp_path / "credentials"
    keys = {"old": _key(1), "new": _key(1)}
    old = CredentialVault(root, CredentialKeyring(keys=keys, active_key_id="old"))
    receipt = _put(old)
    _rewrite_envelope(receipt.path, "key_id", "new")
    reader = CredentialVault(root, CredentialKeyring(keys=keys, active_key_id="new"))

    with pytest.raises(CredentialVaultError):
        reader.resolve(receipt.credential_ref, agent_id="agt_1", app_id="cli_1")


def test_destroy_is_idempotent_and_removes_secret(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault)

    assert vault.destroy(receipt.credential_ref) is True
    assert vault.destroy(receipt.credential_ref) is False
    assert not receipt.path.exists()


def test_destroy_fsyncs_directory_when_credential_is_already_absent(tmp_path, monkeypatch) -> None:
    vault = _vault(tmp_path)
    credential_ref = "cred_" + "0" * 64
    fsynced: list[int] = []
    real_fsync = os.fsync

    def counting_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", counting_fsync)

    assert vault.destroy(credential_ref) is False
    assert fsynced == [vault._root_fd]


def test_vault_rejects_root_symlink(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "credentials"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(CredentialVaultConfigurationError):
        CredentialVault(root, CredentialKeyring(keys={"k1": _key(1)}, active_key_id="k1"))


def test_vault_rejects_leaf_envelope_symlink(tmp_path) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault)
    outside = tmp_path / "outside.json"
    outside.write_bytes(receipt.path.read_bytes())
    outside.chmod(0o600)
    receipt.path.unlink()
    receipt.path.symlink_to(outside)

    with pytest.raises(CredentialVaultError):
        vault.resolve(receipt.credential_ref, agent_id="agt_1", app_id="cli_1")


def test_vault_root_swap_does_not_redirect_operations(tmp_path) -> None:
    root = tmp_path / "credentials"
    vault = CredentialVault(root, CredentialKeyring(keys={"k1": _key(1)}, active_key_id="k1"))
    anchored_root = tmp_path / "anchored-credentials"
    root.rename(anchored_root)
    root.mkdir()

    receipt = _put(vault)
    filename = f"{receipt.credential_ref}.json"

    assert (anchored_root / filename).is_file()
    assert not (root / filename).exists()
    assert vault.resolve(receipt.credential_ref, agent_id="agt_1", app_id="cli_1") == "secret"


def test_vault_close_is_idempotent_and_context_managed(tmp_path) -> None:
    vault = _vault(tmp_path)
    descriptor = vault._root_fd

    with vault as entered:
        assert entered is vault

    with pytest.raises(OSError):
        os.fstat(descriptor)
    vault.close()


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_orphan_scan_rejects_non_integer_schema_versions(tmp_path, schema_version) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault)
    _rewrite_envelope(receipt.path, "schema_version", schema_version)

    with pytest.raises(CredentialVaultError):
        vault.find_orphan_receipts(set())


@pytest.mark.parametrize(
    "field,value",
    [
        ("nonce", "not-base64!"),
        ("nonce", base64.urlsafe_b64encode(b"short").decode()),
        ("ciphertext", "not-base64!"),
        ("ciphertext", ""),
        ("ciphertext_sha256", "A" * 64),
        ("ciphertext_sha256", "0" * 63),
        ("created_at", "not-a-timestamp"),
        ("created_at", "2026-07-12T00:00:00"),
        ("key_id", ""),
        ("agent_id", ""),
        ("app_id", ""),
        ("hire_intent_id", ""),
        ("attempt_id", ""),
    ],
)
def test_orphan_scan_rejects_malformed_envelope_fields(tmp_path, field: str, value: object) -> None:
    vault = _vault(tmp_path)
    receipt = _put(vault)
    _rewrite_envelope(receipt.path, field, value)

    with pytest.raises(CredentialVaultError):
        vault.find_orphan_receipts(set())


def test_vault_rejects_credential_ref_path_traversal(tmp_path) -> None:
    vault = _vault(tmp_path)

    with pytest.raises(CredentialVaultError) as raised:
        vault.resolve("../outside", agent_id="agt_1", app_id="cli_1")

    assert str(raised.value) == "CredentialVaultError:../outside"
