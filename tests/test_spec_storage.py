import json
import os
from types import SimpleNamespace

import src.spec_engine.persistence as persistence
from src.spec_engine.models import SpecProject
from src.spec_engine.persistence import artifact_root_dir, get_state_path, save_engine_state
from src.spec_engine.storage import (
    project_cache_root,
    run_state_path,
    state_path_candidates,
)


def _settings(cache_root: str):
    return SimpleNamespace(
        spec_cache_root=cache_root,
        spec_state_filename=".spec_engine_state.json",
        spec_artifacts_dirname=".spec_engine",
        spec_history_log_filename="history.jsonl",
        spec_state_cycles_tail=50,
        spec_state_work_items_tail=200,
        spec_state_metrics_tail=200,
    )


def test_project_cache_root_mirrors_absolute_project_path(tmp_path):
    settings = _settings(str(tmp_path / "cache"))
    project_root = tmp_path / "workspace" / "repo"

    cache_root = project_cache_root(str(project_root), settings)

    assert cache_root == os.path.join(str(tmp_path / "cache"), *project_root.parts[1:])


def test_default_state_and_artifacts_write_under_cache_root(tmp_path):
    settings = _settings(str(tmp_path / "cache"))
    project_root = tmp_path / "workspace" / "repo"
    project = SpecProject.create(root_path=str(project_root))
    project.requirement = "ship spec cache"

    state_path = save_engine_state(
        project=project,
        settings=settings,
        root_path=str(project_root),
        chat_id="chat-1",
        build_runtime_context_fn=lambda: {"agent_type": "coco"},
        project_to_compact_dict_fn=project.to_dict,
    )

    assert state_path == get_state_path(str(project_root), settings)
    assert state_path.startswith(str(tmp_path / "cache"))
    assert not (project_root / ".spec_engine_state.json").exists()

    per_run_state = run_state_path(str(project_root), settings, project.project_id)
    assert os.path.exists(per_run_state)
    with open(per_run_state, encoding="utf-8") as f:
        data = json.load(f)
    assert data["project"]["requirement"] == "ship spec cache"
    assert artifact_root_dir(str(project_root), settings, project).startswith(str(tmp_path / "cache"))


def test_state_candidates_keep_legacy_project_path_as_fallback(tmp_path):
    settings = _settings(str(tmp_path / "cache"))
    project_root = tmp_path / "repo"

    candidates = state_path_candidates(str(project_root), settings)

    assert candidates[0].startswith(str(tmp_path / "cache"))
    assert candidates[1] == str(project_root / ".spec_engine_state.json")


def test_save_engine_state_writes_run_state_before_project_state(monkeypatch, tmp_path):
    settings = _settings(str(tmp_path / "cache"))
    project_root = tmp_path / "repo"
    project = SpecProject.create(root_path=str(project_root))
    writes: list[str] = []
    original_write = persistence._write_json_atomic

    def spy_write(path: str, data: dict) -> None:
        writes.append(path)
        original_write(path, data)

    monkeypatch.setattr(persistence, "_write_json_atomic", spy_write)

    state_path = save_engine_state(
        project=project,
        settings=settings,
        root_path=str(project_root),
        chat_id="chat-1",
        build_runtime_context_fn=lambda: {},
        project_to_compact_dict_fn=project.to_dict,
    )

    assert writes[0] == run_state_path(str(project_root), settings, project.project_id)
    assert writes[1] == state_path
