"""Security and durability contracts for employee workspace projections."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from src.autonomous.workspace import (
    EmployeeWorkspaceProjector,
    EmployeeWorkspaceSource,
    WorkspaceProjectionError,
)


def _source(**changes) -> EmployeeWorkspaceSource:
    values = {
        "tenant_key": "tenant_1",
        "agent_id": "agt_atlas",
        "name": "Atlas",
        "role": "coder",
        "persona": "careful and evidence driven",
        "personality_traits": ("严谨", "主动沟通"),
        "capabilities": ("coding", "testing", "file_read"),
        "permissions": ("file_read",),
        "tool": "codex",
        "model": "gpt-5",
        "identity_version": 3,
        "projection_sequence": 12,
        "projection_hash": "a" * 64,
    }
    values.update(changes)
    return EmployeeWorkspaceSource(**values)


def test_projector_writes_complete_secret_free_workspace_atomically(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    projector = EmployeeWorkspaceProjector(root)

    first = projector.project(_source())
    before = {
        path.relative_to(root / "agt_atlas"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "agt_atlas").rglob("*")
        if path.is_file()
    }
    second = projector.project(_source())
    after = {
        path.relative_to(root / "agt_atlas"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "agt_atlas").rglob("*")
        if path.is_file()
    }

    required = {
        Path("workspace/AGENTS.md"),
        Path("workspace/IDENTITY.md"),
        Path("workspace/NOW.md"),
        Path("workspace/purpose.md"),
        Path("workspace/schema.md"),
        Path("workspace/wiki/index.md"),
        Path("workspace/wiki/overview.md"),
        Path("workspace/wiki/log.md"),
        Path("workspace/tasks/active.md"),
        Path("workspace/tasks/archive/index.md"),
        Path("workspace/sources/manifest.yaml"),
        Path("runtime/codex-home/AGENTS.md"),
    }
    assert required <= set(after)
    assert before == after
    assert first == second
    agents = (root / "agt_atlas/workspace/AGENTS.md").read_bytes()
    codex_agents = (root / "agt_atlas/runtime/codex-home/AGENTS.md").read_bytes()
    assert len(agents) <= 8192
    assert agents == codex_agents
    assert first.instruction_digest == hashlib.sha256(agents).hexdigest()
    combined = b"\n".join(
        path.read_bytes()
        for path in (root / "agt_atlas/workspace").rglob("*")
        if path.is_file()
    ).lower()
    for forbidden in (b"credential_ref", b"app_secret", b"private raw message", b"sk-secret"):
        assert forbidden not in combined
    for path in (root / "agt_atlas").rglob("*"):
        mode = os.stat(path, follow_symlinks=False).st_mode & 0o777
        assert mode == (0o700 if path.is_dir() else 0o600)


def test_projector_rejects_unsafe_agent_id_and_symlinked_ancestor(tmp_path: Path) -> None:
    projector = EmployeeWorkspaceProjector(tmp_path / "agents")
    with pytest.raises(WorkspaceProjectionError, match="agent_id"):
        projector.project(_source(agent_id="../escape"))
    assert not (tmp_path / "escape").exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    agents = tmp_path / "symlink-agents"
    agents.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        EmployeeWorkspaceProjector(agents).project(_source())
    assert not (outside / "agt_atlas").exists()


def test_projector_rejects_symlinked_employee_directory(tmp_path: Path) -> None:
    agents = tmp_path / "agents"
    outside = tmp_path / "outside"
    agents.mkdir()
    outside.mkdir()
    (agents / "agt_atlas").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        EmployeeWorkspaceProjector(agents).project(_source())

    assert not (outside / "workspace").exists()
