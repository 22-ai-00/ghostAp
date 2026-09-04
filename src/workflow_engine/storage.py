"""Centralized paths for Workflow-owned local artifacts."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WORKFLOW_STORAGE_ROOT = "~/.ghostap/workflow"


def workflow_storage_root(storage_root: str | None = None) -> str:
    """Return the absolute root for all Workflow-owned artifacts."""
    root = storage_root or DEFAULT_WORKFLOW_STORAGE_ROOT
    return os.path.abspath(os.path.expanduser(root))


def workflow_project_storage_root(
    root_path: str,
    storage_root: str | None = None,
) -> str:
    """Mirror one absolute project path below the Workflow storage root.

    Mirroring the full path keeps projects with the same basename isolated
    while leaving the project tree itself untouched.
    """
    abs_project = os.path.abspath(os.path.expanduser(root_path or "."))
    drive, tail = os.path.splitdrive(abs_project)
    parts = [part for part in Path(tail).parts if part not in (os.sep, "")]
    if drive:
        parts.insert(0, drive.rstrip(":"))
    return os.path.join(workflow_storage_root(storage_root), "projects", *parts)


def workflow_scripts_dir(root_path: str, storage_root: str | None = None) -> str:
    """Return the generated-script directory for one project."""
    return os.path.join(
        workflow_project_storage_root(root_path, storage_root),
        "scripts",
    )


def workflow_journal_dir(root_path: str, storage_root: str | None = None) -> str:
    """Return the agent-result journal directory for one project."""
    return os.path.join(
        workflow_project_storage_root(root_path, storage_root),
        "journal",
    )


def workflow_reports_dir(root_path: str, storage_root: str | None = None) -> str:
    """Return the generated-report directory for one project."""
    return os.path.join(
        workflow_project_storage_root(root_path, storage_root),
        "reports",
    )
