"""Filesystem layout helpers for Spec Engine persistence.

Spec runtime artifacts are intentionally kept outside the target repository so
long-running Spec cycles do not pollute git status.  The cache mirrors the
project absolute path under ``~/.cache/ghostAp`` by default:

``/Users/me/project`` -> ``~/.cache/ghostAp/Users/me/project``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

DEFAULT_SPEC_CACHE_ROOT = "~/.cache/ghostAp"
RUN_STATE_FILENAME = "state.json"


def _setting_str(settings, name: str, default: str = "") -> str:
    value = getattr(settings, name, default)
    if isinstance(value, str):
        return value.strip()
    return default


def spec_cache_root(settings=None) -> str:
    configured = _setting_str(settings, "spec_cache_root", "")
    root = configured or DEFAULT_SPEC_CACHE_ROOT
    return os.path.abspath(os.path.expanduser(root))


def project_cache_root(root_path: str, settings=None) -> str:
    abs_project = os.path.abspath(os.path.expanduser(root_path or "."))
    drive, tail = os.path.splitdrive(abs_project)
    parts = [part for part in Path(tail).parts if part not in (os.sep, "")]
    if drive:
        parts.insert(0, drive.rstrip(":"))
    return os.path.join(spec_cache_root(settings), *parts)


def legacy_state_path(root_path: str, settings) -> str:
    return os.path.join(root_path, _setting_str(settings, "spec_state_filename", ".spec_engine_state.json"))


def legacy_artifact_base_dir(root_path: str, settings) -> str:
    return os.path.join(root_path, _setting_str(settings, "spec_artifacts_dirname", ".spec_engine"))


def get_state_path(root_path: str, settings) -> str:
    return os.path.join(project_cache_root(root_path, settings), _setting_str(settings, "spec_state_filename", ".spec_engine_state.json"))


def state_path_candidates(root_path: str, settings) -> list[str]:
    paths = [get_state_path(root_path, settings), legacy_state_path(root_path, settings)]
    result: list[str] = []
    for path in paths:
        if path and path not in result:
            result.append(path)
    return result


def artifact_base_dir(root_path: str, settings) -> str:
    return os.path.join(project_cache_root(root_path, settings), _setting_str(settings, "spec_artifacts_dirname", ".spec_engine"))


def artifact_root_dir(root_path: str, settings, project_id: str) -> str:
    return os.path.join(artifact_base_dir(root_path, settings), project_id or "unknown")


def run_state_path(root_path: str, settings, project_id: str) -> str:
    return os.path.join(artifact_root_dir(root_path, settings, project_id), RUN_STATE_FILENAME)


def iter_artifact_base_dirs(root_path: str, settings) -> Iterable[str]:
    seen: set[str] = set()
    for path in (artifact_base_dir(root_path, settings), legacy_artifact_base_dir(root_path, settings)):
        if path and path not in seen:
            seen.add(path)
            yield path
