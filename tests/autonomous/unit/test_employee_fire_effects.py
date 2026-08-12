from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.autonomous.provisioning.fire_effects import (
    AtomicEmployeeArchive,
    ExecutionQuiesceEffect,
)
from src.autonomous.provisioning.fire_state import DurableFireState


def _state() -> DurableFireState:
    return DurableFireState(
        intent_id="fire_1",
        tenant_key="tenant_1",
        message_id="om_1",
        chat_id="oc_dm",
        requester_principal_id="ou_admin",
        agent_id="agt_alpha",
        employee_name="Alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        credential_ref="cred_alpha",
        drain=False,
    )


def test_atomic_archive_writes_manifest_hashes_and_moves_source_once(tmp_path):
    root = tmp_path / "agents"
    source = root / "agt_alpha"
    source.mkdir(parents=True)
    (source / "history.jsonl").write_text("safe history", encoding="utf-8")
    archive = AtomicEmployeeArchive(root)

    archive.execute(_state())

    destination = root / ".archive" / "agt_alpha"
    assert not source.exists()
    assert archive.observe(_state()) is True
    manifest = json.loads((destination / "archive_manifest.json").read_text())
    assert manifest["files"] == {
        "history.jsonl": hashlib.sha256(b"safe history").hexdigest()
    }
    assert manifest["external_app_disposition"] == "manual_deletion_required"
    assert manifest["credential_destroyed"] is True
    assert manifest["history_date_range"] == {"start": None, "end": None}
    assert manifest["cleanup_disposition"]["credential_destroy"] == "committed"
    archive.execute(_state())
    assert archive.observe(_state()) is True

    (destination / "history.jsonl").write_text("tampered", encoding="utf-8")
    assert archive.observe(_state()) is False


def test_atomic_archive_scrubs_employee_traex_auth_before_move(tmp_path):
    root = tmp_path / "agents"
    source = root / "agt_alpha"
    traex_home = source / "runtime/trae-home"
    auth_file = traex_home / "cli/auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text('{"trae":{"access_token":"secret"}}', encoding="utf-8")
    (traex_home / "AGENTS.md").write_text("# Employee constraints\n", encoding="utf-8")

    archive = AtomicEmployeeArchive(root)
    archive.execute(_state())

    destination = root / ".archive" / "agt_alpha"
    manifest = json.loads((destination / "archive_manifest.json").read_text())
    assert "runtime/trae-home/cli/auth.json" not in manifest["files"]
    assert not (destination / "runtime/trae-home/cli/auth.json").exists()
    assert (destination / "runtime/trae-home/AGENTS.md").is_file()
    assert archive.observe(_state()) is True


def test_atomic_archive_records_empty_archive_when_workspace_was_never_created(tmp_path):
    root = tmp_path / "agents"
    archive = AtomicEmployeeArchive(root)

    archive.execute(_state())

    destination = root / ".archive" / "agt_alpha"
    manifest = json.loads((destination / "archive_manifest.json").read_text())
    assert manifest["files"] == {}
    assert archive.observe(_state()) is True
    archive.execute(_state())
    assert archive.observe(_state()) is True


def test_atomic_archive_records_confirmed_external_disposition_evidence(tmp_path):
    root = tmp_path / "agents"
    state = replace(
        _state(),
        external_disposition_confirmed=True,
        external_disposition_ref="cli_alpha",
        external_disposed_by="ou_admin",
        external_disposed_at="2026-07-15T01:02:03+00:00",
    )
    archive = AtomicEmployeeArchive(root)

    archive.execute(state)

    manifest = json.loads(
        (root / ".archive" / "agt_alpha" / "archive_manifest.json").read_text()
    )
    assert manifest["external_app_disposition"] == "manual_disposition_confirmed"
    assert manifest["external_disposition_ref"] == "cli_alpha"
    assert manifest["external_disposed_by"] == "ou_admin"
    assert manifest["external_disposed_at"] == "2026-07-15T01:02:03+00:00"
    assert archive.observe(state) is True


def test_atomic_archive_rejects_symlink_content(tmp_path):
    root = tmp_path / "agents"
    source = root / "agt_alpha"
    source.mkdir(parents=True)
    target = tmp_path / "outside"
    target.write_text("secret", encoding="utf-8")
    (source / "escape").symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        AtomicEmployeeArchive(root).execute(_state())

    assert source.is_dir()
    assert not (root / ".archive" / "agt_alpha").exists()


def test_execution_quiesce_retires_actor_before_later_fire_effects() -> None:
    retired = []
    runtime = SimpleNamespace(retire_employee=lambda agent_id: retired.append(agent_id))
    coordinator = SimpleNamespace(
        state=SimpleNamespace(attempts={}),
        employee_runtime=runtime,
    )
    ExecutionQuiesceEffect(coordinator, grace_seconds=0).execute(_state())
    assert retired == ["agt_alpha"]


def test_execution_quiesce_drain_never_retires_an_active_actor_after_grace() -> None:
    retired: list[str] = []
    attempt = SimpleNamespace(
        binding=SimpleNamespace(agent_id="agt_alpha"),
        terminal_status="",
    )
    runtime = SimpleNamespace(retire_employee=lambda agent_id: retired.append(agent_id))
    coordinator = SimpleNamespace(
        state=SimpleNamespace(attempts={"att_active": attempt}),
        employee_runtime=runtime,
    )
    state = replace(_state(), drain=True)

    ExecutionQuiesceEffect(coordinator, grace_seconds=0).execute(state)

    assert retired == []
    assert ExecutionQuiesceEffect(coordinator, grace_seconds=0).observe(state) is False


def test_execution_quiesce_drain_closes_terminal_race_before_commit() -> None:
    retired: set[str] = set()
    terminal_reads = iter(("", "completed", "completed", "completed"))

    class _Attempt:
        binding = SimpleNamespace(agent_id="agt_alpha")

        @property
        def terminal_status(self):
            return next(terminal_reads)

    runtime = SimpleNamespace(
        retire_employee=lambda agent_id: retired.add(agent_id),
        is_retired=lambda agent_id: agent_id in retired,
    )
    coordinator = SimpleNamespace(
        state=SimpleNamespace(attempts={"att_race": _Attempt()}),
        employee_runtime=runtime,
    )
    state = replace(_state(), drain=True)
    effect = ExecutionQuiesceEffect(coordinator, grace_seconds=0)

    effect.execute(state)

    assert effect.observe(state) is False
    assert retired == set()

    effect.execute(state)

    assert effect.observe(state) is True
    assert retired == {"agt_alpha"}
