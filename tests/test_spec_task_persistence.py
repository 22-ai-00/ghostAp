import os
import tempfile
import time
from contextlib import contextmanager
from unittest.mock import patch

from src.spec_engine.task_persistence import (
    SpecTaskState,
    delete_task_state,
    generate_task_id,
    list_pending_tasks,
    load_task_state,
    save_task_state,
)


@contextmanager
def patched_task_dirs(primary: str, fallback: str):
    with patch("src.spec_engine.task_persistence.SPEC_TASKS_DIR", primary), patch(
        "src.spec_engine.task_persistence.SPEC_TASKS_DIR_FALLBACK", fallback
    ):
        yield


class TestSpecTaskState:
    def test_to_dict(self):
        state = SpecTaskState(
            task_id="abc12345",
            created_at=1700000000.0,
            requirement="Build a feature",
            project_path="/path/to/project",
            chat_id="chat123",
            agent_type="coco",
            current_cycle=2,
            current_phase="build",
            last_error="timeout",
            retry_count=1,
            status="失败",
            failure_reason="Phase build 失败: timeout",
            models_tried=["model1", "model2"],
            project_snapshot={"key": "value"},
            runtime_context={"engine_name": "Coco", "current_model": "model2"},
        )
        d = state.to_dict()
        assert d["task_id"] == "abc12345"
        assert d["created_at"] == 1700000000.0
        assert d["status"] == "失败"
        assert d["failure_reason"] == "Phase build 失败: timeout"
        assert d["requirement"] == "Build a feature"
        assert d["project_path"] == "/path/to/project"
        assert d["chat_id"] == "chat123"
        assert d["agent_type"] == "coco"
        assert d["current_cycle"] == 2
        assert d["current_phase"] == "build"
        assert d["last_error"] == "timeout"
        assert d["retry_count"] == 1
        assert d["models_tried"] == ["model1", "model2"]
        assert d["project_snapshot"] == {"key": "value"}
        assert d["runtime_context"] == {"engine_name": "Coco", "current_model": "model2"}

    def test_from_dict(self):
        data = {
            "task_id": "xyz99999",
            "created_at": 1600000000.0,
            "requirement": "Test requirement",
            "project_path": "/tmp/test",
            "chat_id": "chat999",
            "agent_type": "claude",
            "current_cycle": 5,
            "current_phase": "review",
            "last_error": "",
            "retry_count": 0,
            "models_tried": ["m1"],
            "project_snapshot": None,
            "runtime_context": {"agent_type": "codex", "engine_name": "Codex", "current_model": "gpt-5.2"},
        }
        state = SpecTaskState.from_dict(data)
        assert state.task_id == "xyz99999"
        assert state.created_at == 1600000000.0
        assert state.requirement == "Test requirement"
        assert state.project_path == "/tmp/test"
        assert state.chat_id == "chat999"
        assert state.agent_type == "claude"
        assert state.current_cycle == 5
        assert state.current_phase == "review"
        assert state.last_error == ""
        assert state.retry_count == 0
        assert state.models_tried == ["m1"]
        assert state.project_snapshot is None
        assert state.runtime_context == {"agent_type": "codex", "engine_name": "Codex", "current_model": "gpt-5.2"}
        assert state.resolved_engine_name() == "Codex"
        assert state.normalized_agent_type() == "codex"
        assert state.resolved_model_name() == "gpt-5.2"

    def test_from_dict_with_defaults(self):
        data = {"task_id": "min", "created_at": 123.0}
        state = SpecTaskState.from_dict(data)
        assert state.task_id == "min"
        assert state.requirement == ""
        assert state.current_cycle == 0
        assert state.models_tried == []
        assert state.project_snapshot is None

    def test_from_dict_back_compat_fills_failure_reason(self):
        """旧快照无 status/failure_reason 时，应从 phase + last_error 补齐。"""
        data = {
            "task_id": "old00001",
            "created_at": 123.0,
            "current_phase": "build",
            "last_error": "Internal error",
        }
        state = SpecTaskState.from_dict(data)
        assert state.status == "失败"
        assert "Phase build 失败: Internal error" in (state.failure_reason or "")

    def test_roundtrip(self):
        original = SpecTaskState(
            task_id="rt123456",
            created_at=time.time(),
            requirement="Roundtrip test",
            project_path="/roundtrip",
            chat_id="c1",
            agent_type="coco",
            current_cycle=3,
            current_phase="task",
            last_error="err",
            retry_count=2,
            models_tried=["a", "b"],
            project_snapshot={"nested": {"data": [1, 2, 3]}},
            runtime_context={"agent_type": "coco", "engine_name": "Coco", "current_model": "b", "models_tried": ["a", "b"]},
        )
        restored = SpecTaskState.from_dict(original.to_dict())
        assert restored.task_id == original.task_id
        assert restored.requirement == original.requirement
        assert restored.project_snapshot == original.project_snapshot
        assert restored.runtime_context == original.runtime_context



class TestGenerateTaskId:
    def test_length(self):
        task_id = generate_task_id()
        assert len(task_id) == 8

    def test_unique(self):
        ids = {generate_task_id() for _ in range(100)}
        assert len(ids) == 100


class TestSaveLoadDelete:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                state = SpecTaskState(
                    task_id="save1234",
                    created_at=time.time(),
                    requirement="Save test",
                    project_path="/save",
                    chat_id="c",
                    agent_type="coco",
                    current_cycle=1,
                    current_phase="spec",
                    last_error="",
                    retry_count=0,
                )
                path = save_task_state(state)
                assert os.path.exists(path)

                loaded = load_task_state("save1234")
                assert loaded is not None
                assert loaded.task_id == "save1234"
                assert loaded.requirement == "Save test"

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                result = load_task_state("nonexistent")
                assert result is None

    def test_load_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                filepath = os.path.join(tmpdir, "corrupt.json")
                with open(filepath, "w") as f:
                    f.write("{invalid json")
                result = load_task_state("corrupt")
                assert result is None

    def test_delete_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                state = SpecTaskState(
                    task_id="del12345",
                    created_at=time.time(),
                    requirement="Delete test",
                    project_path="/del",
                    chat_id="c",
                    agent_type="coco",
                    current_cycle=0,
                    current_phase="spec",
                    last_error="",
                    retry_count=0,
                )
                save_task_state(state)
                result = delete_task_state("del12345")
                assert result is True
                assert load_task_state("del12345") is None

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                result = delete_task_state("nosuchid")
                assert result is False

    def test_atomic_write_cleanup_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                state = SpecTaskState(
                    task_id="atomic12",
                    created_at=time.time(),
                    requirement="Atomic test",
                    project_path="/atomic",
                    chat_id="c",
                    agent_type="claude",
                    current_cycle=0,
                    current_phase="plan",
                    last_error="",
                    retry_count=0,
                )
                save_task_state(state)
                tmp_path = os.path.join(tmpdir, "atomic12.json.tmp")
                assert not os.path.exists(tmp_path)


class TestListPendingTasks:
    def test_list_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                tasks = list_pending_tasks()
                assert tasks == []

    def test_list_nonexistent_directory(self):
        with patched_task_dirs("/nonexistent/path", "/nonexistent/fallback"):
            tasks = list_pending_tasks()
            assert tasks == []

    def test_list_multiple_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                for i in range(3):
                    state = SpecTaskState(
                        task_id=f"list{i:04d}",
                        created_at=time.time() + i,
                        requirement=f"Task {i}",
                        project_path=f"/path{i}",
                        chat_id=f"chat{i}",
                        agent_type="coco",
                        current_cycle=i,
                        current_phase="spec",
                        last_error="",
                        retry_count=0,
                    )
                    save_task_state(state)

                tasks = list_pending_tasks()
                assert len(tasks) == 3
                task_ids = {t.task_id for t in tasks}
                assert task_ids == {"list0000", "list0001", "list0002"}

    def test_list_skips_non_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                state = SpecTaskState(
                    task_id="validtsk",
                    created_at=time.time(),
                    requirement="Valid",
                    project_path="/valid",
                    chat_id="c",
                    agent_type="coco",
                    current_cycle=0,
                    current_phase="spec",
                    last_error="",
                    retry_count=0,
                )
                save_task_state(state)
                with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
                    f.write("ignore me")

                tasks = list_pending_tasks()
                assert len(tasks) == 1
                assert tasks[0].task_id == "validtsk"

    def test_list_skips_corrupted_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patched_task_dirs(tmpdir, os.path.join(tmpdir, "fallback")):
                state = SpecTaskState(
                    task_id="goodfile",
                    created_at=time.time(),
                    requirement="Good",
                    project_path="/good",
                    chat_id="c",
                    agent_type="coco",
                    current_cycle=0,
                    current_phase="spec",
                    last_error="",
                    retry_count=0,
                )
                save_task_state(state)
                with open(os.path.join(tmpdir, "badfile.json"), "w") as f:
                    f.write("not valid json")

                tasks = list_pending_tasks()
                assert len(tasks) == 1
                assert tasks[0].task_id == "goodfile"
