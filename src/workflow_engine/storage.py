"""Centralized paths for Workflow-owned local artifacts."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW_STORAGE_ROOT = "~/.ghostap/workflow"


def workflow_storage_root(storage_root: str | None = None) -> str:
    """Return the absolute root for all Workflow-owned artifacts."""
    root = storage_root or DEFAULT_WORKFLOW_STORAGE_ROOT
    return os.path.realpath(os.path.abspath(os.path.expanduser(root)))


def workflow_project_storage_root(
    root_path: str,
    storage_root: str | None = None,
) -> str:
    """Mirror one absolute project path below the Workflow storage root.

    Mirroring the full path keeps projects with the same basename isolated
    while leaving the project tree itself untouched.
    """
    abs_project = os.path.realpath(os.path.abspath(os.path.expanduser(root_path or ".")))
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


def remove_owned_workflow_artifact(
    path: str | None,
    *,
    root_path: str,
    scripts_dir: str | None = None,
    warn_on_refusal: bool = False,
) -> None:
    """Remove only disposable generated Workflow scripts or immutable execution copies."""
    if not path:
        return

    candidate = os.path.realpath(os.path.abspath(path))
    basename = os.path.basename(candidate)
    resolved_scripts_dir = os.path.realpath(scripts_dir or workflow_scripts_dir(root_path))
    temp_root = os.path.realpath(tempfile.gettempdir())

    generated_source = (
        os.path.dirname(candidate) == resolved_scripts_dir
        and basename.endswith(".js")
        and (basename.startswith("generated-workflow-") or basename == "generated_workflow.js")
    )
    confirmed_copy = (
        basename.startswith(("ghostap-verified-", "ghostap-confirmed-"))
        and basename.endswith(".js")
        and os.path.commonpath((candidate, temp_root)) == temp_root
    )
    if not (generated_source or confirmed_copy):
        if warn_on_refusal:
            logger.warning("[workflow] Refusing to remove non-owned workflow artifact: %s", basename)
        return
    try:
        os.remove(candidate)
    except OSError:
        pass
