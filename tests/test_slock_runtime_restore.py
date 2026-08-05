"""Tests for SlockEngineManager.restore_from_disk — runtime recovery from markers."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

_acp_available = pytest.importorskip("acp", reason="acp package not installed")

from src.slock_engine.manager import (  # noqa: E402
    SlockEngineManager,
    SlockEngineResolutionError,
)
from src.slock_engine.models import SlockChannel  # noqa: E402


def _write_marker(storage_base_path: str, channel_id: str, data: dict) -> str:
    """Helper: write a .slock_channel.json marker under {base}/groups/{channel_id}/."""
    channel_dir = os.path.join(storage_base_path, "groups", channel_id)
    os.makedirs(channel_dir, exist_ok=True)
    marker_path = os.path.join(channel_dir, ".slock_channel.json")
    marker_data = dict(data)
    marker_data.setdefault(
        "root_path",
        os.path.dirname(os.path.dirname(storage_base_path)),
    )
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker_data, f, ensure_ascii=False)
    return marker_path


class TestRestoreFromDiskHappyPath:
    def test_restore_single_engine(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "oc_chat_1", {
            "channel_id": "oc_chat_1",
            "team_name": "Alpha",
            "name": "Alpha Group",
            "activated_at": "2025-01-01T00:00:00Z",
        })

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 1
        assert manager.is_managed_chat("oc_chat_1") is True
        engine = manager.get_activated_engine("oc_chat_1")
        assert engine is not None
        assert engine.channel is not None
        assert engine.channel.channel_id == "oc_chat_1"
        assert engine.channel.team_name == "Alpha"
        assert engine.channel.name == "Alpha Group"

    def test_restore_multiple_engines(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        for i, name in enumerate(["Alpha", "Beta", "Gamma"]):
            _write_marker(storage_base, f"oc_chat_{i}", {
                "channel_id": f"oc_chat_{i}",
                "team_name": name,
                "name": f"{name} Group",
            })

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 3
        for i in range(3):
            assert manager.is_managed_chat(f"oc_chat_{i}") is True
            assert manager.get_activated_engine(f"oc_chat_{i}") is not None

    def test_restore_idempotent(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "oc_chat_1", {
            "channel_id": "oc_chat_1",
            "team_name": "Alpha",
            "name": "Alpha Group",
        })

        manager = SlockEngineManager(storage_base_path=storage_base)
        first = manager.restore_from_disk(root)
        second = manager.restore_from_disk(root)

        assert first == 1
        assert second == 0  # already managed, skipped

    def test_restore_uses_persisted_project_root_instead_of_process_fallback(self, tmp_path):
        fallback_root = tmp_path / "process_cwd"
        project_root = tmp_path / "team_project"
        fallback_root.mkdir()
        project_root.mkdir()
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "oc_chat_1", {
            "channel_id": "oc_chat_1",
            "team_name": "Alpha",
            "name": "Alpha Group",
            "root_path": str(project_root),
        })

        manager = SlockEngineManager(storage_base_path=storage_base)
        assert manager.restore_from_disk(str(fallback_root)) == 1

        engine = manager.get_activated_engine("oc_chat_1")
        assert engine is not None
        assert engine.root_path == str(project_root.resolve())

    def test_untrusted_marker_is_left_durable_for_validated_migration(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        marker_path = _write_marker(
            storage_base,
            "oc_legacy_team",
            {
                "channel_id": "oc_legacy_team",
                "team_name": "Legacy",
                "name": "Legacy Group",
            },
        )
        manager = SlockEngineManager(storage_base_path=storage_base)

        restored = manager.restore_from_disk(
            root,
            managed_group_active=lambda _chat_id: False,
        )

        assert restored == 0
        assert Path(marker_path).is_file()
        assert not list(Path(marker_path).parent.glob(".slock_channel.dissolved.*.json"))

    def test_managed_group_migration_candidates_include_cutover_archive(self, tmp_path):
        project_root = tmp_path / "legacy-project"
        project_root.mkdir()
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        marker_path = Path(
            _write_marker(
                storage_base,
                "oc_legacy_team",
                {
                    "channel_id": "oc_legacy_team",
                    "team_name": "Legacy",
                    "name": "Legacy Group",
                    "owner_id": "ou_owner",
                    "root_path": str(project_root),
                    "activated_at": "2026-07-17T03:38:52Z",
                },
            )
        )
        archived_ns = 1_785_548_041_819_419_550
        archived_path = marker_path.with_name(
            f".slock_channel.dissolved.{archived_ns}.json"
        )
        marker_path.replace(archived_path)
        manager = SlockEngineManager(storage_base_path=storage_base)

        candidates = manager.managed_group_migration_candidates(
            owner_id="ou_owner",
            receiving_bot_ref="cli_main_bot",
            archived_after_ns=1_785_000_000_000_000_000,
        )

        assert candidates == (
            {
                "bound_chat_created_at": 1_784_259_532.0,
                "canonical_root_ref": str(project_root.resolve()),
                "chat_id": "oc_legacy_team",
                "owner_id": "ou_owner",
                "project_id": "team:Legacy",
                "receiving_bot_ref": "cli_main_bot",
                "_archived_path": str(archived_path),
            },
        )


class TestRestoreFromDiskErrorHandling:
    def test_skips_non_lark_chat_id_test_marker(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "test_chat_007", {
            "channel_id": "test_chat_007",
            "team_name": "TestTeam",
            "name": "test",
        })

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 0
        assert manager.is_managed_chat("test_chat_007") is False

    def test_skips_marker_whose_channel_id_does_not_match_directory(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "oc_safe", {
            "channel_id": "oc_other",
            "team_name": "Mismatch",
            "name": "Mismatch",
        })

        manager = SlockEngineManager(storage_base_path=storage_base)

        assert manager.restore_from_disk(root) == 0
        assert manager.is_managed_chat("oc_other") is False

    def test_skips_unavailable_persisted_project_instead_of_using_fallback(self, tmp_path):
        fallback_root = tmp_path / "process_cwd"
        fallback_root.mkdir()
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "oc_missing_project", {
            "channel_id": "oc_missing_project",
            "team_name": "Missing",
            "name": "Missing",
            "root_path": str(tmp_path / "does_not_exist"),
        })
        manager = SlockEngineManager(storage_base_path=storage_base)

        assert manager.restore_from_disk(str(fallback_root)) == 0
        assert manager.is_managed_chat("oc_missing_project") is False

    def test_skips_corrupted_marker(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        channel_dir = os.path.join(storage_base, "groups", "oc_bad")
        os.makedirs(channel_dir)
        with open(os.path.join(channel_dir, ".slock_channel.json"), "w") as f:
            f.write("NOT VALID JSON {{{{")

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 0
        assert manager.is_managed_chat("oc_bad") is False


class TestDeletedChatRetirementRecovery:
    class _Engine:
        def __init__(self, *, deactivate_failures: int = 0, cleanup_failures: int = 0):
            self.root_path = "/project"
            self.channel = object()
            self.deactivate_failures = deactivate_failures
            self.cleanup_failures = cleanup_failures
            self.deactivate_calls = 0
            self.cleanup_calls = 0

        def deactivate(self):
            self.deactivate_calls += 1
            self.channel = None
            if self.deactivate_failures:
                self.deactivate_failures -= 1
                raise RuntimeError("deactivate interrupted")

        def cleanup(self):
            self.cleanup_calls += 1
            if self.cleanup_failures:
                self.cleanup_failures -= 1
                raise RuntimeError("cleanup interrupted")

    @staticmethod
    def _manager_with_engine(tmp_path, engine):
        manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
        key = "oc_deleted:/project"
        manager._engines[key] = engine
        manager._chat_keys["oc_deleted"] = {key}
        manager.register_managed_chat("oc_deleted")
        return manager

    def test_redelivery_retries_engine_whose_deactivate_cleared_channel(self, tmp_path):
        engine = self._Engine(deactivate_failures=1)
        manager = self._manager_with_engine(tmp_path, engine)

        with pytest.raises(RuntimeError, match="deactivate interrupted"):
            manager.retire_deleted_chat("oc_deleted")

        assert manager.list_engines("oc_deleted") == [engine]
        assert engine.channel is None
        manager.retire_deleted_chat("oc_deleted")
        assert manager.list_engines("oc_deleted") == []
        assert manager.is_managed_chat("oc_deleted") is False
        assert engine.deactivate_calls == 2

    def test_redelivery_retries_engine_whose_remove_cleanup_failed(self, tmp_path):
        engine = self._Engine(cleanup_failures=1)
        manager = self._manager_with_engine(tmp_path, engine)

        with pytest.raises(RuntimeError, match="cleanup interrupted"):
            manager.retire_deleted_chat("oc_deleted")

        assert manager.list_engines("oc_deleted") == [engine]
        manager.retire_deleted_chat("oc_deleted")
        assert manager.list_engines("oc_deleted") == []
        assert engine.cleanup_calls == 2

    def test_retirement_blocks_and_rejects_concurrent_reactivation(
        self,
        tmp_path,
        monkeypatch,
    ):
        manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
        archive_entered = threading.Event()
        release_archive = threading.Event()
        retirement_done = threading.Event()
        activation_done = threading.Event()
        activation_errors: list[BaseException] = []

        def blocking_archive(_chat_id: str):
            archive_entered.set()
            assert release_archive.wait(timeout=2)
            return None

        monkeypatch.setattr(manager, "archive_managed_chat_marker", blocking_archive)

        def retire():
            manager.retire_deleted_chat("oc_deleted")
            retirement_done.set()

        def reactivate():
            try:
                manager.get_or_create_activated(
                    "oc_deleted",
                    str(tmp_path),
                    SlockChannel(channel_id="oc_deleted"),
                    engine_name="Slock",
                )
            except BaseException as exc:  # captured for the test thread
                activation_errors.append(exc)
            finally:
                activation_done.set()

        retirement_thread = threading.Thread(target=retire)
        activation_thread = threading.Thread(target=reactivate)
        retirement_thread.start()
        assert archive_entered.wait(timeout=2)
        activation_thread.start()

        assert activation_done.wait(timeout=0.1) is False
        release_archive.set()
        retirement_thread.join(timeout=2)
        activation_thread.join(timeout=2)

        assert retirement_done.is_set()
        assert activation_done.is_set()
        assert len(activation_errors) == 1
        assert isinstance(activation_errors[0], SlockEngineResolutionError)
        assert manager.list_engines("oc_deleted") == []
        assert manager.is_managed_chat("oc_deleted") is False


class TestActivatedBindingPublication:
    class _Engine:
        engine_name = "Coco"
        _agent_type = "coco"
        _model_name = None
        is_running = False

        def __init__(self, *, activation_error: BaseException | None = None):
            self.channel = None
            self.activation_error = activation_error
            self.activation_entered = threading.Event()
            self.release_activation = threading.Event()
            self.deactivate_calls = 0
            self.cleanup_calls = 0

        def activate_channel(self, channel):
            self.channel = channel
            self.activation_entered.set()
            assert self.release_activation.wait(timeout=2)
            if self.activation_error is not None:
                raise self.activation_error

        def deactivate(self):
            self.deactivate_calls += 1
            self.channel = None

        def cleanup(self):
            self.cleanup_calls += 1

    @staticmethod
    def _manager_with_created_engine(tmp_path, monkeypatch, engine):
        manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
        monkeypatch.setattr(manager, "_create_engine", lambda **_kwargs: engine)
        return manager

    def test_activation_failure_rolls_back_uncommitted_engine(self, tmp_path, monkeypatch):
        engine = self._Engine(activation_error=OSError("marker write failed"))
        engine.release_activation.set()
        manager = self._manager_with_created_engine(tmp_path, monkeypatch, engine)

        with pytest.raises(OSError, match="marker write failed"):
            manager.get_or_create_activated(
                "oc_failed",
                str(tmp_path),
                SlockChannel(channel_id="oc_failed"),
            )

        assert manager.get_activated_engine("oc_failed") is None
        assert manager.list_engines("oc_failed") == []
        assert manager.is_managed_chat("oc_failed") is False
        assert engine.channel is None
        assert engine.deactivate_calls == 1
        assert engine.cleanup_calls == 1

    def test_reader_cannot_observe_binding_before_activation_commits(self, tmp_path, monkeypatch):
        engine = self._Engine()
        manager = self._manager_with_created_engine(tmp_path, monkeypatch, engine)
        activation_done = threading.Event()
        reader_done = threading.Event()
        reader_results: list[object] = []

        def activate():
            manager.get_or_create_activated(
                "oc_pending",
                str(tmp_path),
                SlockChannel(channel_id="oc_pending"),
            )
            activation_done.set()

        def read():
            reader_results.append(manager.get_activated_engine("oc_pending"))
            reader_done.set()

        activation_thread = threading.Thread(target=activate)
        reader_thread = threading.Thread(target=read)
        activation_thread.start()
        assert engine.activation_entered.wait(timeout=2)
        reader_thread.start()
        try:
            assert reader_done.wait(timeout=0.1) is False
        finally:
            engine.release_activation.set()
        activation_thread.join(timeout=2)
        reader_thread.join(timeout=2)

        assert activation_done.is_set()
        assert reader_done.is_set()
        assert reader_results == [engine]
        assert manager.is_managed_chat("oc_pending") is True


class TestRestoreFromDiskAdditionalErrorHandling:

    def test_skips_missing_channel_id(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        _write_marker(storage_base, "oc_no_id", {
            "team_name": "NoId",
            "name": "Missing channel_id",
        })

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 0
        assert manager.is_managed_chat("oc_no_id") is False

    def test_no_slock_dir_returns_zero(self, tmp_path):
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(str(tmp_path))
        assert restored == 0

    def test_empty_slock_dir_returns_zero(self, tmp_path):
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        os.makedirs(os.path.join(storage_base, "groups"))
        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(str(tmp_path))
        assert restored == 0

    def test_skips_dir_without_marker(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")
        os.makedirs(os.path.join(storage_base, "groups", "oc_no_marker"))

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 0

    def test_mixed_valid_and_corrupted(self, tmp_path):
        root = str(tmp_path)
        storage_base = str(tmp_path / "ghostap_config" / "slock")

        # Valid marker
        _write_marker(storage_base, "oc_good", {
            "channel_id": "oc_good",
            "team_name": "Good",
            "name": "Good Group",
        })

        # Corrupted marker
        channel_dir = os.path.join(storage_base, "groups", "oc_bad")
        os.makedirs(channel_dir)
        with open(os.path.join(channel_dir, ".slock_channel.json"), "w") as f:
            f.write("{broken")

        manager = SlockEngineManager(storage_base_path=storage_base)
        restored = manager.restore_from_disk(root)

        assert restored == 1
        assert manager.is_managed_chat("oc_good") is True
        assert manager.is_managed_chat("oc_bad") is False
