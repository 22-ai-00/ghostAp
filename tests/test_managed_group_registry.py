"""Task 0.9 contracts for the durable managed-group trust registry."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, tzinfo

import pytest

from src.trust.models import ManagedGroupOrigin, ManagedGroupStatus
from src.trust.registry import (
    ManagedGroupConflictError,
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
        "migration_dispositions",
        "provision_intents",
        "revoke_intents",
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


def test_two_registry_instances_merge_mutations_without_lost_updates(tmp_path):
    path = tmp_path / "managed-groups.json"
    first = ManagedGroupRegistry(path)
    stale = ManagedGroupRegistry(path)

    _activate(first, chat_id="oc_first")
    _activate(stale, chat_id="oc_second")

    replayed = ManagedGroupRegistry(path)
    assert {record.chat_id for record in replayed.managed_groups()} == {
        "oc_first",
        "oc_second",
    }


def test_stale_registry_instance_cannot_overwrite_a_durable_tombstone(tmp_path):
    path = tmp_path / "managed-groups.json"
    writer = ManagedGroupRegistry(path)
    _activate(writer, chat_id="oc_retired")
    stale = ManagedGroupRegistry(path)

    writer.tombstone("oc_retired")
    _activate(stale, chat_id="oc_new")

    replayed = ManagedGroupRegistry(path)
    assert replayed.active_record("oc_retired") is None
    assert replayed.record("oc_retired").status is ManagedGroupStatus.TOMBSTONED
    assert replayed.active_record("oc_new") is not None


def test_post_replace_fsync_failure_recovers_authoritative_commit(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "managed-groups.json"
    registry = ManagedGroupRegistry(path)
    registry.begin_provision(
        provision_id="provision:oc_managed",
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        created_at=NOW,
    )
    original_fsync = os.fsync
    calls = 0

    def fail_parent_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        original_fsync(fd)

    monkeypatch.setattr("src.trust.registry.os.fsync", fail_parent_fsync)

    record, grant = registry.activate(
        provision_id="provision:oc_managed",
        chat_id="oc_managed",
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        backend_binding_ids=("codex",),
    )

    assert registry.active_record("oc_managed") == record
    assert registry.grant_for_chat("oc_managed") == grant
    assert ManagedGroupRegistry(path).active_record("oc_managed") == record


def test_registry_rejects_boolean_version(tmp_path):
    path = tmp_path / "managed-groups.json"
    path.write_text(
        json.dumps(
            {
                "groups": {},
                "provision_intents": {},
                "revision": 0,
                "schema": "ghostap.managed_groups",
                "version": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RegistryCorruptionError):
        ManagedGroupRegistry(path)


def test_registry_rejects_symlink_target_or_parent(tmp_path):
    real_path = tmp_path / "real.json"
    ManagedGroupRegistry(real_path)
    target_link = tmp_path / "target-link.json"
    target_link.symlink_to(real_path)
    with pytest.raises(RegistryCorruptionError):
        ManagedGroupRegistry(target_link)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RegistryCorruptionError):
        ManagedGroupRegistry(parent_link / "registry.json")


class _NaiveOffset(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


def test_runtime_datetime_requires_concrete_utc_offset(tmp_path):
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")

    with pytest.raises(ValueError, match="timezone-aware"):
        registry.begin_provision(
            provision_id="bad-time",
            owner_id="ou_owner",
            origin=ManagedGroupOrigin.GHOSTAP_CREATED,
            receiving_bot_ref="cli_main_bot",
            project_id="project-1",
            canonical_root_ref="/srv/project-1",
            created_at=datetime(2026, 7, 31, tzinfo=_NaiveOffset()),
        )


def test_provision_intent_binds_one_remote_chat_across_restart(tmp_path):
    path = tmp_path / "managed-groups.json"
    registry = ManagedGroupRegistry(path)
    registry.begin_provision(
        provision_id="new-chat:project-1",
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        created_at=NOW,
    )

    assert registry.bind_provision_chat("new-chat:project-1", "oc_created") == "oc_created"
    replayed = ManagedGroupRegistry(path)
    assert replayed.provision_chat_id("new-chat:project-1") == "oc_created"
    assert replayed.bind_provision_chat("new-chat:project-1", "oc_created") == "oc_created"
    with pytest.raises(ManagedGroupConflictError, match="different chat"):
        replayed.bind_provision_chat("new-chat:project-1", "oc_other")


def test_pending_revoke_fails_closed_across_restart_until_resolved(tmp_path):
    path = tmp_path / "managed-groups.json"
    registry = ManagedGroupRegistry(path)
    active, _ = _activate(registry)

    registry.begin_revoke("oc_managed", requested_at=NOW)
    assert registry.active_record("oc_managed") is None
    assert ManagedGroupRegistry(path).pending_revokes() == ("oc_managed",)

    registry.cancel_revoke("oc_managed")
    assert registry.active_record("oc_managed") == active
    registry.begin_revoke("oc_managed", requested_at=NOW)
    tombstone = registry.complete_revoke("oc_managed")
    assert tombstone.status is ManagedGroupStatus.TOMBSTONED
    assert registry.pending_revokes() == ()
