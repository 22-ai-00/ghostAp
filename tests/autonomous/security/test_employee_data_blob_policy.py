from __future__ import annotations

import base64
import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from src.autonomous.data.keyring import (
    EmployeeDataConfigurationError,
    EmployeeDataKeyring,
    build_employee_data_storage,
)
from src.autonomous.data.policy import (
    build_document_labels,
    build_history_labels,
    validate_blob_ref_labels,
)
from src.autonomous.journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobFormatError,
    BlobIntegrityError,
    BlobStore,
)


def _key(byte: int) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).decode()


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        autonomous_data_keys=SecretStr(
            json.dumps({"version": 1, "keys": {"old": _key(1), "new": _key(2)}})
        ),
        autonomous_data_active_key_id="new",
        autonomous_data_blob_dir=str(tmp_path / "blobs"),
    )


def test_data_keyring_parses_rotation_set_without_repr_or_error_leaks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    keyring = EmployeeDataKeyring.from_settings(settings)
    assert keyring.active_key_id == "new"
    assert set(keyring.keys) == {"old", "new"}
    assert _key(1) not in repr(keyring)

    settings.autonomous_data_active_key_id = "missing"
    with pytest.raises(EmployeeDataConfigurationError) as raised:
        EmployeeDataKeyring.from_settings(settings)
    assert str(raised.value) == "EmployeeDataConfigurationError"
    assert _key(1) not in str(raised.value)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        '{"version":2,"keys":{"k":"value"}}',
        '{"version":1,"keys":{"k":"short"}}',
        '{"version":1,"keys":{"k":"%s"},"extra":1}' % _key(1),
        '{"version":1,"keys":{"k":"%s","k":"%s"}}' % (_key(1), _key(2)),
    ],
)
def test_data_keyring_rejects_malformed_or_ambiguous_json(raw: str, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.autonomous_data_keys = SecretStr(raw)
    settings.autonomous_data_active_key_id = "k"
    with pytest.raises(EmployeeDataConfigurationError):
        EmployeeDataKeyring.from_settings(settings)


def test_composition_factory_uses_active_and_old_keys(tmp_path: Path) -> None:
    storage = build_employee_data_storage(_settings(tmp_path))
    with storage:
        ref = storage.blob_store.stage_and_publish(
            b"secret payload",
            build_history_labels("tenant_1", "principal_1", "hist_" + "a" * 64),
            storage.active_key_id,
        )
        assert ref.key_ref == "new"
        assert storage.blob_store.read(ref) == b"secret payload"
    assert storage.blob_store.closed is True


def test_composition_factory_canonicalizes_only_the_user_home_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(linked_home))
    settings = _settings(tmp_path)
    settings.autonomous_data_blob_dir = "~/.ghostap/data-blobs"

    storage = build_employee_data_storage(settings)
    try:
        assert storage.blob_store.root == real_home / ".ghostap/data-blobs"
    finally:
        storage.close()


def test_composition_factory_still_rejects_a_symlink_inside_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_home = tmp_path / "real-home"
    store_parent = real_home / ".ghostap"
    store_parent.mkdir(parents=True)
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store_parent / "data-blobs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HOME", str(linked_home))
    settings = _settings(tmp_path)
    settings.autonomous_data_blob_dir = "~/.ghostap/data-blobs"

    with pytest.raises(EmployeeDataConfigurationError):
        build_employee_data_storage(settings)


def test_exact_label_policy_rejects_extra_cross_resource_and_secret_keys(
    tmp_path: Path,
) -> None:
    expected = build_history_labels(
        "tenant_1", "principal_1", "hist_" + "a" * 64
    )
    assert set(expected) == {
        "tenant_key",
        "owner_principal_id",
        "classification",
        "purpose",
        "resource_id",
        "schema_version",
    }
    document = build_document_labels(
        tenant_key="tenant_1",
        owner_principal_id="principal_1",
        document_id="data_0123456789abcdef",
        kind="memory_summary",
    )
    assert document["purpose"] == "memory_summary"

    store = BlobStore(
        tmp_path / "blobs",
        AesGcmEncryptionProvider(lambda _ref: bytes([3]) * 32),
    )
    try:
        ref = store.stage_and_publish(b"payload", expected, "k1")
        validate_blob_ref_labels(ref, expected)
        with pytest.raises(ValueError):
            validate_blob_ref_labels(ref, {**expected, "tenant_key": "tenant_2"})
        with pytest.raises(ValueError):
            validate_blob_ref_labels(ref, {**expected, "api_secret": "leak"})
    finally:
        store.close()


@pytest.mark.parametrize(
    ("purpose", "resource_id"),
    [
        ("execution_history", "data_0123456789abcdef"),
        ("memory_summary", "hist_" + "a" * 64),
    ],
)
def test_label_validation_binds_purpose_to_resource_shape(
    tmp_path: Path,
    purpose: str,
    resource_id: str,
) -> None:
    labels = {
        "tenant_key": "tenant_1",
        "owner_principal_id": "principal_1",
        "classification": "restricted",
        "purpose": purpose,
        "resource_id": resource_id,
        "schema_version": "1",
    }
    store = _store(tmp_path / "blobs")
    try:
        ref = store.stage_and_publish(b"payload", labels, "k1")
        with pytest.raises(ValueError, match="resource_id"):
            validate_blob_ref_labels(ref, labels)
    finally:
        store.close()


def _store(root: Path) -> BlobStore:
    return BlobStore(root, AesGcmEncryptionProvider(lambda _ref: bytes([4]) * 32))


def test_blob_store_root_rename_does_not_redirect_reads_or_writes(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    store = _store(root)
    first = store.stage_and_publish(b"first", {"tenant": "one"}, "k1")
    moved = tmp_path / "moved"
    root.rename(moved)
    root.mkdir()

    second = store.stage_and_publish(b"second", {"tenant": "one"}, "k1")

    assert store.read(first) == b"first"
    assert (moved / f"{second.blob_id}.blob").exists()
    assert list(root.iterdir()) == []
    store.close()


def test_blob_store_rejects_symlink_and_non_0600_leaf(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    store = _store(root)
    ref = store.stage_and_publish(b"payload", {"tenant": "one"}, "k1")
    target = root / f"{ref.blob_id}.blob"
    target.chmod(0o644)
    with pytest.raises(BlobIntegrityError):
        store.read(ref)
    target.chmod(0o600)
    outside = tmp_path / "outside"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(BlobIntegrityError):
        store.read(ref)
    store.close()


def test_blob_store_rejects_symlink_root_without_chmodding_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    root = tmp_path / "blobs"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(Exception):
        _store(root)

    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_blob_store_schema_version_is_literal_int(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    store = _store(root)
    ref = store.stage_and_publish(b"payload", {"tenant": "one"}, "k1")
    path = root / f"{ref.blob_id}.blob"
    envelope = json.loads(path.read_bytes())
    envelope["schema_version"] = True
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    rewritten_hash = hashlib.sha256(raw).hexdigest()
    (root / f"{rewritten_hash}.blob").write_bytes(raw)
    (root / f"{rewritten_hash}.blob").chmod(0o600)
    tampered = type(ref)(
        blob_hash=rewritten_hash,
        payload_hash=ref.payload_hash,
        labels_hash=ref.labels_hash,
        key_ref=ref.key_ref,
        size=ref.size,
        labels=ref.labels,
    )
    with pytest.raises(BlobFormatError):
        store.read(tampered)
    store.close()


def test_blob_iteration_and_quarantine_are_owned_and_durable(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    store = _store(root)
    first = store.stage_and_publish(b"one", {"tenant": "one"}, "k1")
    second = store.stage_and_publish(b"two", {"tenant": "one"}, "k1")
    (root / "unrelated.txt").write_text("keep")

    assert store.iter_blob_ids() == tuple(sorted((first.blob_id, second.blob_id)))
    quarantined = store.quarantine_blob(first.blob_id)
    assert quarantined.name == f"{first.blob_id}.blob"
    assert store.iter_blob_ids() == (second.blob_id,)
    assert (root / "quarantine" / quarantined.name).exists()
    assert (root / "unrelated.txt").read_text() == "keep"
    store.close()


def test_blob_store_close_is_idempotent_and_redacted(tmp_path: Path) -> None:
    secret = bytes([5]) * 32
    store = BlobStore(tmp_path / "blobs", AesGcmEncryptionProvider(lambda _ref: secret))
    assert secret.hex() not in repr(store)
    store.close()
    store.close()
    assert store.closed is True
    with pytest.raises(BlobIntegrityError):
        store.iter_blob_ids()
