"""Task 0.9 contracts for the durable managed-group trust registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.trust.models import ManagedGroupOrigin, ManagedGroupStatus
from src.trust.registry import (
    ManagedGroupRegistry,
    RegistryCorruptionError,
)

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def _activate(
    registry: ManagedGroupRegistry,
    *,
    chat_id: str = "oc_managed",
    bot_ref: str = "cli_main_bot",
    origin: ManagedGroupOrigin = ManagedGroupOrigin.GHOSTAP_CREATED,
):
    intent_id = f"provision:{chat_id}"
    registry.begin_provision(
        provision_id=intent_id,
        owner_id="ou_owner",
        origin=origin,
        receiving_bot_ref=bot_ref,
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        created_at=NOW,
    )
    return registry.activate(
        provision_id=intent_id,
        chat_id=chat_id,
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        backend_binding_ids=("codex",),
    )


def test_registry_replays_before_ingress_and_preserves_one_grant(tmp_path):
    path = tmp_path / "managed-groups.json"
    first = ManagedGroupRegistry(path)
    record, grant = _activate(first)

    replayed = ManagedGroupRegistry(path)

    assert replayed.active_record("oc_managed") == record
    assert replayed.grant_for_chat("oc_managed") == grant
    assert replayed.managed_groups() == (record,)
    assert replayed.project_grants() == (grant,)
    assert record.project_grant_id == grant.grant_id


def test_marker_name_or_allowed_chat_cannot_forge_managed_group(tmp_path):
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    stale_project = {
        "bound_chat_id": "oc_forged",
        "bound_chat_created_at": NOW.timestamp(),
        "root_path": "/srv/project-1",
        "allowed_chat_ids": ["oc_forged"],
    }
    stale_slock_marker = {
        "channel_id": "oc_forged",
        "name": "Trusted [Slock]",
        "owner_id": "ou_owner",
    }

    assert registry.active_record("oc_forged") is None
    assert registry.import_candidate(stale_project, validator=lambda _: False) is None
    assert registry.import_candidate(stale_slock_marker, validator=lambda _: True) is None
    assert registry.import_candidate(
        {
            "bound_chat_created_at": NOW.timestamp(),
            "canonical_root_ref": "/srv/project-1",
            "chat_id": None,
            "owner_id": "ou_owner",
            "project_id": "project-1",
            "receiving_bot_ref": "cli_main_bot",
        },
        validator=lambda _: True,
    ) is None
    assert registry.active_record("oc_forged") is None


def test_tombstoned_group_cannot_resurrect_from_stale_project_or_slock_marker(tmp_path):
    path = tmp_path / "managed-groups.json"
    registry = ManagedGroupRegistry(path)
    active, _ = _activate(registry)
    tombstone = registry.tombstone("oc_managed")

    assert tombstone.status is ManagedGroupStatus.TOMBSTONED
    assert tombstone.revision > active.revision
    assert registry.active_record("oc_managed") is None
    assert registry.import_candidate(
        {
            "chat_id": "oc_managed",
            "owner_id": "ou_owner",
            "receiving_bot_ref": "cli_main_bot",
            "project_id": "project-1",
            "canonical_root_ref": "/srv/project-1",
            "bound_chat_created_at": NOW.timestamp(),
        },
        validator=lambda _: True,
    ) is None
    replayed = ManagedGroupRegistry(path)
    assert replayed.record("oc_managed") == tombstone
    assert replayed.active_record("oc_managed") is None
    assert replayed.grant_for_chat("oc_managed") is None


def test_expected_employee_rotation_updates_revision_without_prompt(tmp_path):
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    before, before_grant = _activate(registry, bot_ref="cli_employee_v1")

    after = registry.rotate_receiving_bot(
        chat_id="oc_managed",
        expected_bot_ref="cli_employee_v1",
        new_bot_ref="cli_employee_v2",
    )
    repeated = registry.rotate_receiving_bot(
        chat_id="oc_managed",
        expected_bot_ref="cli_employee_v1",
        new_bot_ref="cli_employee_v2",
    )

    assert after.receiving_bot_ref == "cli_employee_v2"
    assert after.revision > before.revision
    assert repeated == after
    assert registry.grant_for_chat("oc_managed") == before_grant


def test_owner_p2p_can_adopt_existing_group_once_without_per_message_enrollment(tmp_path):
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    kwargs = {
        "chat_id": "oc_existing",
        "owner_id": "ou_owner",
        "receiving_bot_ref": "cli_main_bot",
        "project_id": "project-7",
        "canonical_root_ref": "/srv/project-7",
        "created_at": NOW,
    }

    first_record, first_grant = registry.adopt_existing(
        **kwargs,
        validator=lambda facts: facts["chat_id"] == "oc_existing",
    )
    second_record, second_grant = registry.adopt_existing(
        **kwargs,
        validator=lambda facts: facts["chat_id"] == "oc_existing",
    )

    assert first_record.origin is ManagedGroupOrigin.OWNER_ADOPTED
    assert second_record == first_record
    assert second_grant == first_grant
    assert registry.managed_groups() == (first_record,)
    assert registry.project_grants() == (first_grant,)


def test_registry_strict_version_and_corruption_fail_closed(tmp_path):
    path = tmp_path / "managed-groups.json"
    path.write_text('{"schema":"ghostap.managed_groups","version":999}', encoding="utf-8")
    with pytest.raises(RegistryCorruptionError):
        ManagedGroupRegistry(path)

    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RegistryCorruptionError):
        ManagedGroupRegistry(path)


def test_registry_snapshot_uses_exact_versioned_schema(tmp_path):
    path = tmp_path / "managed-groups.json"
    registry = ManagedGroupRegistry(path)
    _activate(registry)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "groups",
        "provision_intents",
        "revision",
        "schema",
        "version",
    }
    assert payload["schema"] == "ghostap.managed_groups"
    assert payload["version"] == 1


def test_dangling_provision_retry_keeps_original_timestamp(tmp_path):
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    kwargs = {
        "provision_id": "new-chat:project-1",
        "owner_id": "ou_owner",
        "origin": ManagedGroupOrigin.GHOSTAP_CREATED,
        "receiving_bot_ref": "cli_main_bot",
        "project_id": "project-1",
        "canonical_root_ref": "/srv/project-1",
    }
    registry.begin_provision(**kwargs, created_at=NOW)

    assert registry.begin_provision(
        **kwargs,
        created_at=NOW + timedelta(seconds=30),
    ) == "new-chat:project-1"
