import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.project.context import (
    ProjectContext,
    ProjectStatus,
)
from src.project.manager import ProjectManager
from src.project.mapper import MessageLinker, MessageProjectMapper


class TestProjectContext:
    def test_create_project_context(self):
        ctx = ProjectContext(
            project_id="test_project",
            project_name="Test Project",
            root_path="/tmp/test",
        )

        assert ctx.project_id == "test_project"
        assert ctx.project_name == "Test Project"
        assert ctx.status == ProjectStatus.IDLE
        assert ctx.coco_mode is False
        assert len(ctx.conversation_history) == 0

    def test_touch_updates_last_active(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
        )

        old_active = ctx.last_active
        import time

        time.sleep(0.01)
        ctx.touch()

        assert ctx.last_active > old_active

    def test_add_conversation(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
            max_history_size=3,
        )

        ctx.add_conversation("user", "Hello")
        ctx.add_conversation("assistant", "Hi there")

        assert len(ctx.conversation_history) == 2
        assert ctx.conversation_history[0].role == "user"
        assert ctx.conversation_history[0].content == "Hello"

    def test_conversation_history_limit(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
            max_history_size=3,
        )

        for i in range(5):
            ctx.add_conversation("user", f"Message {i}")

        assert len(ctx.conversation_history) == 3
        assert ctx.conversation_history[0].content == "Message 2"

    def test_set_coco_mode(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
        )

        ctx.set_coco_mode(True, "session_123", 5)

        assert ctx.coco_mode is True
        assert ctx.coco_session_snapshot is not None
        assert ctx.coco_session_snapshot.session_id == "session_123"
        assert ctx.coco_session_snapshot.query_count == 5

    def test_set_traex_mode(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
        )

        ctx.set_traex_mode(True, "session_traex", 7)

        assert ctx.traex_mode is True
        assert ctx.traex_session_snapshot is not None
        assert ctx.traex_session_snapshot.session_id == "session_traex"
        assert ctx.traex_session_snapshot.query_count == 7

    def test_get_status_emoji(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
            emoji_prefix="🟢",
        )

        ctx.status = ProjectStatus.IDLE
        assert ctx.get_status_emoji() == "⚪"

        ctx.status = ProjectStatus.ACTIVE
        assert ctx.get_status_emoji() == "🟢"

        ctx.status = ProjectStatus.BUSY
        assert ctx.get_status_emoji() == "🟡"

    def test_to_snapshot_and_from_snapshot(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test Project",
            root_path="/tmp/test",
            theme_color="blue",
            emoji_prefix="🔵",
        )
        ctx.set_coco_mode(True, "session_456", 10)
        ctx.set_traex_mode(True, "session_traex", 7)

        snapshot = ctx.to_snapshot()

        assert snapshot["project_id"] == "test"
        assert snapshot["project_name"] == "Test Project"
        assert snapshot["coco_session_snapshot"]["session_id"] == "session_456"
        assert snapshot["traex_session_snapshot"]["session_id"] == "session_traex"

        restored = ProjectContext.from_snapshot(snapshot)

        assert restored.project_id == ctx.project_id
        assert restored.project_name == ctx.project_name
        assert restored.theme_color == ctx.theme_color
        assert restored.coco_session_snapshot.session_id == "session_456"
        assert restored.traex_session_snapshot.session_id == "session_traex"

    def test_snapshot_compare_and_clear_is_atomic_with_replacement(self):
        ctx = ProjectContext(
            project_id="test",
            project_name="Test",
            root_path="/tmp/test",
        )
        old_read = threading.Event()
        allow_old_read = threading.Event()

        class BlockingSnapshot:
            query_count = 1
            last_query = ""
            is_resumable = True

            @property
            def session_id(self):
                old_read.set()
                allow_old_read.wait(timeout=2)
                return "old-session"

        ctx.coco_session_snapshot = BlockingSnapshot()
        cleared: list[bool] = []

        clear_worker = threading.Thread(
            target=lambda: cleared.append(
                ctx.clear_programming_snapshot_if_matches(
                    "coco",
                    "old-session",
                )
            )
        )
        clear_worker.start()
        assert old_read.wait(timeout=1)

        replacement_worker = threading.Thread(
            target=lambda: ctx.set_programming_mode(
                "coco",
                True,
                "replacement-session",
                0,
            )
        )
        replacement_worker.start()
        allow_old_read.set()
        clear_worker.join(timeout=2)
        replacement_worker.join(timeout=2)

        assert not clear_worker.is_alive()
        assert not replacement_worker.is_alive()
        assert cleared == [True]
        assert ctx.coco_session_snapshot is not None
        assert ctx.coco_session_snapshot.session_id == "replacement-session"


class TestProjectManager:
    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "projects.json")
            yield storage_path

    @pytest.fixture
    def project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_create_project(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        success, msg, project = manager.create_project(
            project_id="test_proj",
            project_name="Test Project",
            root_path=project_dir,
            chat_id="chat_123",
        )

        assert success is True
        assert project is not None
        assert project.project_id == "test_proj"
        assert project.project_name == "Test Project"

    def test_persist_project_context_flushes_snapshot_tombstone(
        self,
        temp_storage,
        project_dir,
    ):
        manager = ProjectManager(storage_path=temp_storage)
        success, _, project = manager.create_project(
            project_id="persist-tombstone",
            project_name="Persist Tombstone",
            root_path=project_dir,
            chat_id="chat-1",
        )
        assert success and project is not None
        project.set_programming_mode("coco", True, "old-session", 1)
        assert manager.persist_project_context(project) is True
        assert project.clear_programming_snapshot_if_matches(
            "coco",
            "old-session",
        )
        assert manager.persist_project_context(project) is True

        restored = ProjectManager(storage_path=temp_storage)
        restored_project = restored.get_project_for_chat("persist-tombstone")
        assert restored_project is not None
        assert restored_project.coco_session_snapshot is None

    def test_persist_project_context_reports_disk_failure(
        self,
        temp_storage,
        project_dir,
    ):
        manager = ProjectManager(storage_path=temp_storage)
        success, _, project = manager.create_project(
            project_id="persist-failure",
            project_name="Persist Failure",
            root_path=project_dir,
            chat_id="chat-1",
        )
        assert success and project is not None

        with patch.object(
            manager,
            "_write_atomic",
            side_effect=OSError("disk unavailable"),
        ):
            assert manager.persist_project_context(project) is False

    def test_create_duplicate_project(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("test", "Test", project_dir)
        success, msg, project = manager.create_project("test", "Test 2", project_dir)

        assert success is False
        assert "已存在" in msg

    def test_get_project(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test", "Test", project_dir)

        project = manager.get_project_for_diagnostics("test")

        assert project is not None
        assert project.project_id == "test"

    def test_get_all_projects(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("proj1", "Project 1", project_dir)
        manager.create_project("proj2", "Project 2", project_dir)

        projects = manager.get_all_projects()

        assert len(projects) == 2

    def test_set_active_project(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test", "Test", project_dir)

        success, msg = manager.set_active_project("chat_123", "test")

        assert success is True

        active = manager.get_active_project("chat_123")
        assert active is not None
        assert active.project_id == "test"
        assert active.status == ProjectStatus.ACTIVE

    def test_set_active_project_same_project_persists_touch(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test", "Test", project_dir)
        manager.set_active_project("chat_123", "test")
        first_active = manager.get_project_for_diagnostics("test").last_active

        time.sleep(0.01)
        success, _ = manager.set_active_project("chat_123", "test")

        assert success is True
        second_active = manager.get_project_for_diagnostics("test").last_active
        assert second_active > first_active

        reloaded = ProjectManager(storage_path=temp_storage).get_project_for_diagnostics("test")
        assert reloaded is not None
        assert reloaded.last_active == second_active

    def test_close_project(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test", "Test", project_dir)

        success, msg = manager.close_project("test")

        assert success is True
        assert manager.get_project_for_diagnostics("test") is None

    def test_find_project_by_name(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("my_app", "My Application", project_dir)

        project = manager.find_project_by_name("My Application")
        assert project is not None
        assert project.project_id == "my_app"

        project = manager.find_project_by_name("my_app")
        assert project is not None

        project = manager.find_project_by_name("app")
        assert project is not None

    def test_persistence(self, temp_storage, project_dir):
        manager1 = ProjectManager(storage_path=temp_storage)
        manager1.create_project("test", "Test", project_dir)

        manager2 = ProjectManager(storage_path=temp_storage)
        project = manager2.get_project_for_diagnostics("test")

        assert project is not None
        assert project.project_name == "Test"

    def test_persistence_corrupted_file_backup(self, temp_storage):
        storage_path = Path(temp_storage)
        storage_path.write_text("{invalid json", encoding="utf-8")

        manager = ProjectManager(storage_path=temp_storage)
        assert manager.get_all_projects() == []

        corrupt_files = list(storage_path.parent.glob(f"{storage_path.name}.corrupt.*"))
        assert len(corrupt_files) == 1

    def test_get_all_projects_sorted_by_recent(self, temp_storage, project_dir):
        import time

        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("proj_a", "Project A", project_dir)
        time.sleep(0.01)
        manager.create_project("proj_b", "Project B", project_dir)
        time.sleep(0.01)
        manager.create_project("proj_c", "Project C", project_dir)

        projects = manager.get_all_projects(sort_by_recent=True)
        assert len(projects) == 3
        assert projects[0].project_id == "proj_c"
        assert projects[1].project_id == "proj_b"
        assert projects[2].project_id == "proj_a"

        manager.get_project_for_diagnostics("proj_a").touch()
        projects = manager.get_all_projects(sort_by_recent=True)
        assert projects[0].project_id == "proj_a"

    def test_get_all_projects_unsorted(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("proj_a", "Project A", project_dir)
        manager.create_project("proj_b", "Project B", project_dir)

        projects = manager.get_all_projects(sort_by_recent=False)
        assert len(projects) == 2

    def test_find_project_by_path(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test_proj", "Test Project", project_dir)

        found = manager.find_project_by_path(project_dir)
        assert found is not None
        assert found.project_id == "test_proj"

        found = manager.find_project_by_path("/nonexistent/path")
        assert found is None

    def test_find_project_by_path_with_tilde(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("test_proj", "Test Project", project_dir)

        abs_path = os.path.abspath(project_dir)
        found = manager.find_project_by_path(abs_path)
        assert found is not None
        assert found.project_id == "test_proj"

        found = manager.find_project_by_path(project_dir + "/")
        assert found is not None
        assert found.project_id == "test_proj"

    def test_get_or_create_project_for_path_existing(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("existing", "Existing Project", project_dir)

        project, is_new = manager.get_or_create_project_for_path(project_dir, "chat_123")

        assert is_new is False
        assert project.project_id == "existing"
        assert manager.get_active_project("chat_123") == project

    def test_get_or_create_project_for_path_existing_persists_touch_without_chat(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("existing", "Existing Project", project_dir)
        project = manager.get_project_for_diagnostics("existing")
        old_active = project.last_active

        time.sleep(0.01)
        found, is_new = manager.get_or_create_project_for_path(project_dir)

        assert is_new is False
        assert found.project_id == "existing"
        assert found.last_active > old_active

        reloaded = ProjectManager(storage_path=temp_storage).get_project_for_diagnostics("existing")
        assert reloaded is not None
        assert reloaded.last_active == found.last_active

    def test_get_or_create_project_for_path_new(self, temp_storage):
        manager = ProjectManager(storage_path=temp_storage)

        with tempfile.TemporaryDirectory() as tmpdir:
            project, is_new = manager.get_or_create_project_for_path(tmpdir, "chat_456")

            assert is_new is True
            assert project.root_path == os.path.abspath(tmpdir)
            assert project.project_name == os.path.basename(tmpdir)

    def test_get_or_create_project_auto_naming(self, temp_storage):
        manager = ProjectManager(storage_path=temp_storage)

        with tempfile.TemporaryDirectory(prefix="my-test-project-") as tmpdir:
            project, is_new = manager.get_or_create_project_for_path(tmpdir)

            assert is_new is True
            expected_name = os.path.basename(tmpdir)
            assert project.project_name == expected_name
            assert "_" in project.project_id or project.project_id == expected_name.lower().replace("-", "_").replace(
                " ", "_"
            )

    def test_get_or_create_project_id_collision(self, temp_storage):
        manager = ProjectManager(storage_path=temp_storage)

        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                os.makedirs(os.path.join(tmpdir1, "myapp"), exist_ok=True)
                os.makedirs(os.path.join(tmpdir2, "myapp"), exist_ok=True)

                path1 = os.path.join(tmpdir1, "myapp")
                path2 = os.path.join(tmpdir2, "myapp")

                proj1, _ = manager.get_or_create_project_for_path(path1)
                proj2, _ = manager.get_or_create_project_for_path(path2)

                assert proj1.project_id != proj2.project_id
                assert proj2.project_id.startswith("myapp_")

    def test_search_projects(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("frontend_app", "Frontend Application", project_dir)
        manager.create_project("backend_api", "Backend API", project_dir)
        manager.create_project("mobile_app", "Mobile App", project_dir)

        results = manager.search_projects("app")
        assert len(results) == 2
        project_ids = [p.project_id for p in results]
        assert "frontend_app" in project_ids
        assert "mobile_app" in project_ids

        results = manager.search_projects("API")
        assert len(results) == 1
        assert results[0].project_id == "backend_api"

        results = manager.search_projects("nonexistent")
        assert len(results) == 0

    def test_search_projects_by_path(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test_proj", "Test Project", project_dir)

        partial_path = os.path.basename(project_dir)
        results = manager.search_projects(partial_path)

        assert len(results) >= 1
        assert any(p.project_id == "test_proj" for p in results)

    def test_search_projects_sorted_by_recent(self, temp_storage, project_dir):
        import time

        manager = ProjectManager(storage_path=temp_storage)

        manager.create_project("app_old", "Old App", project_dir)
        time.sleep(0.01)
        manager.create_project("app_new", "New App", project_dir)

        results = manager.search_projects("app")
        assert len(results) == 2
        assert results[0].project_id == "app_new"
        assert results[1].project_id == "app_old"

    def test_validate_project_path_valid(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("valid_proj", "Valid Project", project_dir)

        valid, path = manager.validate_project_path("valid_proj")

        assert valid is True
        assert path == project_dir

    def test_validate_project_path_nonexistent_project(self, temp_storage):
        manager = ProjectManager(storage_path=temp_storage)

        valid, msg = manager.validate_project_path("nonexistent")

        assert valid is False
        assert "不存在" in msg

    def test_validate_project_path_invalid_path(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("valid_proj", "Valid Project", project_dir)

        project = manager.get_project_for_diagnostics("valid_proj")
        project.root_path = "/nonexistent/path/12345"

        valid, msg = manager.validate_project_path("valid_proj")

        assert valid is False
        assert "路径不存在" in msg

    def test_update_working_dir(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test_proj", "Test Project", project_dir)

        subdir = os.path.join(project_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)

        success, result = manager.update_working_dir("test_proj", subdir)

        assert success is True
        assert result == subdir

        project = manager.get_project_for_diagnostics("test_proj")
        assert project.working_dir == subdir

    def test_update_working_dir_relative_path(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test_proj", "Test Project", project_dir)

        subdir = os.path.join(project_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)

        success, result = manager.update_working_dir("test_proj", "subdir")

        assert success is True
        assert result == subdir

    def test_update_working_dir_nonexistent(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)
        manager.create_project("test_proj", "Test Project", project_dir)

        success, msg = manager.update_working_dir("test_proj", "/nonexistent/path")

        assert success is False
        assert "不存在" in msg

    def test_update_working_dir_persists(self, temp_storage, project_dir):
        manager1 = ProjectManager(storage_path=temp_storage)
        manager1.create_project("test_proj", "Test Project", project_dir)

        subdir = os.path.join(project_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        manager1.update_working_dir("test_proj", subdir)

        manager2 = ProjectManager(storage_path=temp_storage)
        project = manager2.get_project_for_diagnostics("test_proj")

        assert project.working_dir == subdir

    def test_switch_project_preserves_working_dir(self, temp_storage, project_dir):
        manager = ProjectManager(storage_path=temp_storage)

        proj1_dir = os.path.join(project_dir, "proj1")
        proj2_dir = os.path.join(project_dir, "proj2")
        os.makedirs(proj1_dir, exist_ok=True)
        os.makedirs(proj2_dir, exist_ok=True)

        manager.create_project("proj1", "Project 1", proj1_dir, chat_id="chat1")
        manager.create_project("proj2", "Project 2", proj2_dir)

        subdir1 = os.path.join(proj1_dir, "src")
        os.makedirs(subdir1, exist_ok=True)
        manager.update_working_dir("proj1", subdir1)

        manager.set_active_project("chat1", "proj2")

        proj1 = manager.get_project_for_diagnostics("proj1")
        assert proj1.working_dir == subdir1

        proj2 = manager.get_project_for_diagnostics("proj2")
        assert proj2.working_dir == proj2_dir


class TestMessageProjectMapper:
    def test_register_and_get(self):
        mapper = MessageProjectMapper()

        mapper.register("msg_123", "project_abc")

        project_id = mapper.get_project_id("msg_123")
        assert project_id == "project_abc"

    def test_get_nonexistent(self):
        mapper = MessageProjectMapper()

        project_id = mapper.get_project_id("nonexistent")
        assert project_id is None

    def test_ttl_expiration(self):
        mapper = MessageProjectMapper(ttl=0)

        mapper.register("msg_123", "project_abc")

        import time

        time.sleep(0.01)

        project_id = mapper.get_project_id("msg_123")
        assert project_id is None

    def test_max_size_limit(self):
        mapper = MessageProjectMapper(max_size=3)

        mapper.register("msg_1", "proj_1")
        mapper.register("msg_2", "proj_2")
        mapper.register("msg_3", "proj_3")
        mapper.register("msg_4", "proj_4")

        assert len(mapper) <= 3
        assert mapper.get_project_id("msg_1") is None
        assert mapper.get_project_id("msg_4") == "proj_4"

    def test_clear(self):
        mapper = MessageProjectMapper()

        mapper.register("msg_1", "proj_1")
        mapper.register("msg_2", "proj_2")

        mapper.clear()

        assert len(mapper) == 0


class TestMessageLinker:
    def test_link_and_query(self):
        linker = MessageLinker(ttl=60, max_size=100)

        origin = "om_origin"
        request_id = "req_123"
        linker.register_origin(origin, request_id=request_id, chat_id="chat_1", project_id="p1")

        reply1 = "om_reply_1"
        reply2 = "om_reply_2"
        linker.link_reply(origin, reply1)
        linker.link_reply(origin, reply2)

        run1 = "run_aaa"
        run2 = "run_bbb"
        linker.link_task(origin, run1)
        linker.link_task(origin, run2)

        data = linker.query(origin)
        assert data is not None
        assert data["origin_message_id"] == origin
        assert data["request_id"] == request_id
        assert reply1 in data["reply_message_ids"]
        assert reply2 in data["reply_message_ids"]
        assert run1 in data["task_run_ids"]
        assert run2 in data["task_run_ids"]

        assert linker.query(reply1)["origin_message_id"] == origin
        assert linker.query(run2)["origin_message_id"] == origin
        assert linker.query(request_id)["origin_message_id"] == origin

    def test_trusted_origin_registration_is_atomic_and_rejects_partial_conflict(self):
        linker = MessageLinker(ttl=60, max_size=100)
        linker.register_origin("om_partial", request_id="req_partial", chat_id="chat_dm")

        assert linker.register_trusted_origin_if_absent(
            "om_partial",
            chat_id="chat_dm",
            sender_id="ou_admin",
            chat_type="p2p",
        ) is False
        assert linker.query("om_partial")["sender_id"] is None

        assert linker.register_trusted_origin_if_absent(
            "om_fresh",
            chat_id="chat_dm",
            sender_id="ou_admin",
            chat_type="p2p",
        ) is True
        assert linker.register_trusted_origin_if_absent(
            "om_fresh",
            chat_id="chat_other",
            sender_id="ou_admin",
            chat_type="p2p",
        ) is False

    def test_trusted_origin_tenant_attachment_is_one_atomic_cas(self):
        linker = MessageLinker(ttl=60, max_size=100)
        assert linker.register_trusted_origin_if_absent(
            "om_team",
            chat_id="oc_team",
            sender_id="ou_requester",
            chat_type="group",
        ) is True

        assert linker.register_trusted_origin_with_tenant(
            "om_team",
            chat_id="oc_team",
            sender_id="ou_requester",
            chat_type="group",
            tenant_key="tenant-a",
        ) is True
        assert linker.register_trusted_origin_with_tenant(
            "om_team",
            chat_id="oc_other",
            sender_id="ou_other",
            chat_type="group",
            tenant_key="tenant-b",
        ) is False

        origin = linker.query("om_team")
        assert origin is not None
        assert origin["chat_id"] == "oc_team"
        assert origin["sender_id"] == "ou_requester"
        assert origin["tenant_key"] == "tenant-a"

    def test_refresh_order_allows_later_expired_origin_cleanup(
        self,
        monkeypatch,
    ):
        now = 0.0
        monkeypatch.setattr("src.project.mapper.time.time", lambda: now)
        linker = MessageLinker(ttl=10, max_size=100)

        linker.register_origin("om_a", request_id="req_a")
        now = 1.0
        linker.register_origin("om_b", request_id="req_b")
        linker.link_reply("om_b", "om_reply_b")
        linker.link_task("om_b", "run_b")

        now = 9.0
        linker.register_origin("om_a", request_id="req_a")
        now = 12.0
        linker.register_origin("om_c", request_id="req_c")

        assert linker.query("om_a") is not None
        assert linker.resolve_origin(reply_message_id="om_reply_b") is None
        assert linker.resolve_origin(run_id="run_b") is None
        assert linker.resolve_origin(request_id="req_b") is None

    def test_refresh_order_capacity_evicts_oldest_and_clears_reverse_indexes(
        self,
        monkeypatch,
    ):
        now = 0.0
        monkeypatch.setattr("src.project.mapper.time.time", lambda: now)
        linker = MessageLinker(ttl=60, max_size=2)

        linker.register_origin("om_a", request_id="req_a")
        now = 1.0
        linker.register_origin("om_b", request_id="req_b")
        linker.link_reply("om_b", "om_reply_b")
        linker.link_task("om_b", "run_b")

        now = 2.0
        linker.register_origin("om_a", request_id="req_a")
        now = 3.0
        linker.register_origin("om_c", request_id="req_c")

        assert linker.query("om_a") is not None
        assert linker.resolve_origin(reply_message_id="om_reply_b") is None
        assert linker.resolve_origin(run_id="run_b") is None
        assert linker.resolve_origin(request_id="req_b") is None


class TestEvictionCallbackOutsideLock:
    """AC-R01: on_eviction fires outside _lock — callback can safely use ProjectManager."""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "projects.json")

    @pytest.fixture
    def project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_callback_can_call_get_all_projects_without_deadlock(
        self, temp_storage, project_dir
    ):
        """on_eviction callback calling get_all_projects() must not deadlock."""
        import threading
        from unittest.mock import MagicMock, patch

        settings = MagicMock()
        settings.max_allowed_chat_ids = 2
        settings.max_evicted_cache = 200

        with patch("src.config.get_settings", return_value=settings):
            manager = ProjectManager(storage_path=temp_storage)

            callback_result = {}
            callback_completed = threading.Event()

            def on_eviction_cb(evicted_cid, proj_name, proj_id):
                # This would deadlock if on_eviction fires inside _lock
                result = manager.get_all_projects(chat_id=None)
                callback_result["projects"] = result
                callback_completed.set()

            manager.on_eviction = on_eviction_cb

            manager.create_project("p1", "Proj1", project_dir, chat_id="chatA")
            manager.set_active_project("chatA", "p1")
            manager.set_active_project("chatB", "p1")
            # chatC should evict chatA (limit=2, owner=chatA but LRU oldest)
            manager.set_active_project("chatC", "p1")

            # Must complete within 5s, otherwise it's a deadlock
            assert callback_completed.wait(timeout=5), "on_eviction callback deadlocked"
            assert "projects" in callback_result


class TestEvictionCleansActiveProject:
    """AC-R02: LRU eviction must remove orphan _active_project entries."""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "projects.json")

    @pytest.fixture
    def project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_evicted_chat_removed_from_active_project(
        self, temp_storage, project_dir
    ):
        from unittest.mock import MagicMock, patch

        settings = MagicMock()
        settings.max_allowed_chat_ids = 2
        settings.max_evicted_cache = 200

        with patch("src.config.get_settings", return_value=settings):
            manager = ProjectManager(storage_path=temp_storage)
            manager.create_project("p1", "Proj1", project_dir, chat_id="owner")

            manager.set_active_project("owner", "p1")
            manager.set_active_project("chatB", "p1")
            # chatC evicts 'owner' (oldest non-owner would be 'chatB' if owner != owner
            # Actually: owner_chat_id is 'owner', so chatB is evicted)
            manager.set_active_project("chatC", "p1")

            # chatB was evicted — must not be in _active_project
            assert manager._active_project.get("chatB") is None
            # chatC and owner should still be present
            assert manager._active_project.get("chatC") == "p1"

    def test_eviction_does_not_clobber_new_binding(
        self, temp_storage, project_dir
    ):
        """Conditional pop must not remove _active_project entry that was
        re-bound to a *different* project between add_chat_id and pop.

        Scenario:
          - p1 (limit=2) has [owner, chatB].
          - chatB is also active on p2 (_active_project["chatB"] = "p2")
            because another thread just switched it.
          - chatC joins p1, evicting chatB from p1's allowed_chat_ids.
          - The conditional pop must NOT remove _active_project["chatB"]
            because it now points to "p2", not "p1".
        """
        from unittest.mock import MagicMock, patch

        settings = MagicMock()
        settings.max_allowed_chat_ids = 2
        settings.max_evicted_cache = 200

        with patch("src.config.get_settings", return_value=settings):
            manager = ProjectManager(storage_path=temp_storage)
            proj2_dir = os.path.join(project_dir, "p2")
            os.makedirs(proj2_dir, exist_ok=True)

            manager.create_project("p1", "Proj1", project_dir, chat_id="owner")
            manager.create_project("p2", "Proj2", proj2_dir)

            manager.set_active_project("owner", "p1")
            manager.set_active_project("chatB", "p1")

            # Simulate another thread switching chatB to p2 right before
            # eviction pop executes.  We poke _active_project directly
            # (still inside the same thread — the lock is reentrant).
            manager._active_project["chatB"] = "p2"

            # chatC joins p1 → evicts chatB from p1's allowed list
            manager.set_active_project("chatC", "p1")

            # chatB's binding to p2 must survive the eviction
            assert manager._active_project.get("chatB") == "p2"
            assert manager._active_project.get("chatC") == "p1"


class TestConcurrentSetActiveNoTOCTOU:
    """Stress-test: concurrent set_active_project must not produce ghost
    entries or clobber valid bindings in _active_project."""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "projects.json")

    def test_concurrent_set_active_no_toctou(self, temp_storage):
        import threading
        from unittest.mock import MagicMock, patch

        settings = MagicMock()
        settings.max_allowed_chat_ids = 3
        settings.max_evicted_cache = 200

        with patch("src.config.get_settings", return_value=settings):
            manager = ProjectManager(storage_path=temp_storage)

            # Create 4 projects, each in its own directory
            proj_dirs = []
            for i in range(4):
                d = os.path.join(os.path.dirname(temp_storage), f"proj{i}")
                os.makedirs(d, exist_ok=True)
                proj_dirs.append(d)
                manager.create_project(
                    f"p{i}", f"Proj{i}", d, chat_id=f"owner{i}"
                )

            num_threads = 12
            barrier = threading.Barrier(num_threads)
            errors: list[str] = []

            def worker(idx: int):
                try:
                    barrier.wait(timeout=5)
                    chat_id = f"chat_{idx}"
                    project_id = f"p{idx % 4}"
                    manager.set_active_project(chat_id, project_id)
                except Exception as e:
                    errors.append(f"thread-{idx}: {e}")

            threads = [
                threading.Thread(target=worker, args=(i,))
                for i in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Thread errors: {errors}"

            # Invariant 1: every chat_id in _active_project must point to
            # an existing project
            for cid, pid in manager._active_project.items():
                assert pid in ("p0", "p1", "p2", "p3"), (
                    f"Ghost entry: {cid} -> {pid}"
                )

            # Invariant 2: every worker's chat_id that succeeded should
            # appear in _active_project
            for i in range(num_threads):
                cid = f"chat_{i}"
                pid = manager._active_project.get(cid)
                # It's possible the binding was evicted by a later thread,
                # but if present it must point to a valid project.
                if pid is not None:
                    assert pid in ("p0", "p1", "p2", "p3")


# ---------------------------------------------------------------------------
# AC-R01 / AC-R03: _is_visible + create_project type correctness
# ---------------------------------------------------------------------------

class TestIsVisibleNoFrozensetAllocation:
    """AC-R01: _is_visible must NOT call _chat_id_set (zero allocation)."""

    def test_is_visible_uses_allowed_chat_ids_directly(self):
        ctx = ProjectContext(
            project_id="p1",
            project_name="P1",
            root_path="/tmp/test_vis",
            owner_chat_id="chatA",
            allowed_chat_ids=OrderedDict([("chatA", time.time())]),
        )
        # Monkey-patch _chat_id_set to detect unwanted calls
        original = ctx._chat_id_set
        call_count = 0
        def spy():
            nonlocal call_count
            call_count += 1
            return original()
        ctx._chat_id_set = spy

        assert ProjectManager._is_visible(ctx, "chatA") is True
        assert ProjectManager._is_visible(ctx, "chatX") is False
        assert call_count == 0, f"_chat_id_set was called {call_count} times; expected 0"


class TestCreateProjectPassesOrderedDict:
    """AC-R03: create_project must pass OrderedDict, not list[tuple]."""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "projects.json")

    def test_allowed_chat_ids_is_ordered_dict(self, temp_storage):
        settings = MagicMock()
        settings.max_allowed_chat_ids = 50
        settings.max_evicted_cache = 200

        with patch("src.config.get_settings", return_value=settings):
            manager = ProjectManager(storage_path=temp_storage)
            d = os.path.join(os.path.dirname(temp_storage), "proj_od")
            os.makedirs(d, exist_ok=True)
            ok, _, ctx = manager.create_project("od_test", "OD Test", d, chat_id="chatOD")
            assert ok
            # Must be OrderedDict, not converted from list by __post_init__
            assert type(ctx.allowed_chat_ids) is OrderedDict
            assert "chatOD" in ctx.allowed_chat_ids
