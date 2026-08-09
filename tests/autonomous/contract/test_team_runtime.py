"""Contract tests for TeamRuntime coordinate parsing and binding behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.autonomous.team.runtime import TeamRuntime, TeamRuntimeResolutionError


class TrackingSessionHost:
    """Capture accidental runtime/session host side effects."""

    def __init__(self) -> None:
        self.activation_calls = 0
        self.retirement_calls = 0
        self.closed = 0

    def open_employee_session(self, *_args, **_kwargs) -> None:
        self.activation_calls += 1
        raise AssertionError("TeamRuntime should not open employee sessions")

    def run_agent_session(self, *_args, **_kwargs) -> None:
        self.activation_calls += 1
        raise AssertionError("TeamRuntime should not run employee sessions")

    def retire_employee(self, *_args, **_kwargs) -> None:
        self.retirement_calls += 1

    def close(self) -> None:
        self.closed += 1


def _resolve_chat_team_root(coordinate: str, base: Path) -> str:
    chat_id, sep, team_id = coordinate.partition("/")
    if sep != "/":
        raise TeamRuntimeResolutionError("invalid chat/team coordinate")
    return str((base / chat_id / team_id).resolve())


def _identity(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def test_chat_team_coordinate_parsing_is_stable(tmp_path: Path) -> None:
    (tmp_path / "chat_alpha" / "team_one").mkdir(parents=True)
    host = TrackingSessionHost()
    runtime = TeamRuntime(
        project_root_resolver=lambda coordinate: _resolve_chat_team_root(
            coordinate,
            tmp_path,
        ),
        owner_resolver=lambda coordinate: coordinate.rsplit("/", 1)[-1],
        session_host=host,
    )

    binding = runtime.resolve_employee_engine(chat_id="chat_alpha/team_one")

    root = str((tmp_path / "chat_alpha" / "team_one").resolve())
    assert binding.chat_id == "chat_alpha/team_one"
    assert binding.engine_identity == _identity("team", "chat_alpha/team_one")
    assert binding.root_identity == _identity("root", root)
    assert binding.canonical_root == root
    assert binding.engine is host


def test_get_activated_engine_binds_session_host_and_resolved_root(tmp_path: Path) -> None:
    (tmp_path / "chat_beta" / "team_two").mkdir(parents=True)
    host = TrackingSessionHost()
    runtime = TeamRuntime(
        project_root_resolver=lambda coordinate: _resolve_chat_team_root(
            coordinate,
            tmp_path,
        ),
        owner_resolver=lambda coordinate: coordinate.rsplit("/", 1)[-1],
        session_host=host,
    )

    activated = runtime.get_activated_engine(chat_id="chat_beta/team_two")

    assert activated.channel.owner_id == "team_two"
    assert activated.root_path == str((tmp_path / "chat_beta" / "team_two").resolve())
    assert host.activation_calls == 0
    runtime.close()
    assert host.closed == 1


def test_missing_chat_team_mapping_is_rejected() -> None:
    runtime = TeamRuntime(
        project_root_resolver=lambda _coordinate: "",
    )

    with pytest.raises(TeamRuntimeResolutionError, match="project root is unavailable"):
        runtime.resolve_employee_engine(chat_id="chat_unknown/team_unknown")


def test_ambiguous_chat_team_mapping_is_rejected() -> None:
    runtime = TeamRuntime(
        project_root_resolver=lambda _coordinate: ("path_a", "path_b"),
    )

    with pytest.raises(TeamRuntimeResolutionError, match="project root is unavailable"):
        runtime.resolve_employee_engine(chat_id="chat_alpha/team_alpha")


def test_activation_context_does_not_activate_or_retire_group_engines(tmp_path: Path) -> None:
    (tmp_path / "chat_gamma" / "team_three").mkdir(parents=True)
    host = TrackingSessionHost()
    runtime = TeamRuntime(
        project_root_resolver=lambda coordinate: _resolve_chat_team_root(
            coordinate,
            tmp_path,
        ),
        owner_resolver=lambda coordinate: f"owner_{coordinate.rsplit('/', 1)[-1]}",
        session_host=host,
    )

    with runtime.employee_activation_guard(chat_id="chat_gamma/team_three") as binding:
        assert binding.engine is host
        assert binding.engine_identity == _identity("team", "chat_gamma/team_three")
        assert host.activation_calls == 0
        assert host.retirement_calls == 0

    runtime.close()
    assert host.closed == 1
