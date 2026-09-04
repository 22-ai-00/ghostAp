"""Contracts for centralized, project-isolated Workflow storage."""

from __future__ import annotations

import os
from pathlib import Path

from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.journal import WorkflowJournal
from src.workflow_engine.models import AgentCallResult
from src.workflow_engine.storage import (
    workflow_journal_dir,
    workflow_project_storage_root,
    workflow_reports_dir,
    workflow_scripts_dir,
)


def test_workflow_artifacts_share_one_project_isolated_root(tmp_path) -> None:
    project = tmp_path / "workspace" / "service"
    storage_root = tmp_path / "home" / ".ghostap" / "workflow"

    project_root = Path(
        workflow_project_storage_root(str(project), str(storage_root))
    )

    assert Path(workflow_scripts_dir(str(project), str(storage_root))).parent == project_root
    assert Path(workflow_journal_dir(str(project), str(storage_root))).parent == project_root
    assert Path(workflow_reports_dir(str(project), str(storage_root))).parent == project_root
    assert os.path.commonpath((project_root, project)) != str(project)


def test_same_project_name_at_different_paths_does_not_collide(tmp_path) -> None:
    storage_root = tmp_path / "storage"
    first = tmp_path / "team-a" / "service"
    second = tmp_path / "team-b" / "service"

    first_root = workflow_project_storage_root(str(first), str(storage_root))
    second_root = workflow_project_storage_root(str(second), str(storage_root))

    assert first_root != second_root
    assert first_root.endswith("team-a/service")
    assert second_root.endswith("team-b/service")


def test_workflow_writes_no_dot_ghostap_directory_in_project(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    script_path = Path(WorkflowHandler._new_workflow_script_path(str(project)))
    journal = WorkflowJournal(str(project), "run-1")
    key = journal.compute_key("prompt", "codex", "model")
    journal.store(key, AgentCallResult(output="done"))
    engine = WorkflowEngine(chat_id="chat-1", root_path=str(project))

    assert script_path.parent == Path(workflow_scripts_dir(str(project)))
    assert Path(engine._state_dir()) == Path(workflow_project_storage_root(str(project)))
    assert Path(journal._journal_dir).parent == Path(workflow_journal_dir(str(project)))
    assert not (project / ".ghostap").exists()
