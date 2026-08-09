from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent_session import factory as session_factory
from src.agent_session.factory import employee_session_environment
from src.autonomous.gateway.env_scope import (
    EmployeeProcessEnvironmentMaterial,
    build_employee_backend_env,
    build_employee_process_env,
)
from src.autonomous.runtime.employee_session import EmployeeSessionBootstrap
from src.autonomous.workforce.identity import AgentIdentity


@pytest.mark.parametrize(
    "backend",
    ("codex", "coco", "traex", "claude", "gemini"),
)
def test_every_backend_receives_same_explicit_employee_bootstrap(
    tmp_path: Path,
    backend: str,
) -> None:
    employee = tmp_path / "agents/agt_boot"
    workspace = employee / "workspace"
    workspace.mkdir(parents=True)
    instruction = b"# Employee: Atlas\n\nUse durable task state.\n"
    (workspace / "AGENTS.md").write_bytes(instruction)
    (employee / "runtime/codex-home").mkdir(parents=True)
    agent = AgentIdentity(
        agent_id="agt_boot",
        name="Atlas",
        agent_type=backend,
        model_name="model",
        workspace_path=str(workspace),
        security_profile="employee_v1",
        permissions=["file_read"],
        capabilities=["file_read"],
    )

    bootstrap = EmployeeSessionBootstrap.from_agent(
        tenant_key="tenant_1",
        agent=agent,
        project_root=str(tmp_path / "project"),
        identity_version=7,
    )

    assert bootstrap.instruction_digest == hashlib.sha256(instruction).hexdigest()
    assert bootstrap.session_key.identity_version == 7
    assert bootstrap.session_key.instruction_digest == bootstrap.instruction_digest
    assert bootstrap.session_key.effort == agent.reasoning_effort
    assert bootstrap.session_key.backend == backend
    assert bootstrap.instruction_digest in bootstrap.wrap_prompt("do work")
    assert "agt_boot" in bootstrap.wrap_prompt("do work")
    assert bootstrap.workspace_root == str(workspace.resolve())


def test_codex_home_is_explicit_not_inherited(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_home = tmp_path / "runtime/codex-home"
    env = build_employee_process_env(
        {"PATH": "/usr/bin", "CODEX_HOME": "/manager/codex"},
        employee_home=str(home),
        codex_home=str(codex_home),
    )
    assert env["HOME"] == str(home)
    assert env["CODEX_HOME"] == str(codex_home)


def test_traex_home_projects_auth_and_employee_constraints(tmp_path: Path) -> None:
    from src.autonomous.gateway import env_scope

    employee_home = tmp_path / "employee"
    auth_file = tmp_path / "manager-trae/cli/auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_payload = b'{"auth_mode":"trae","trae":{"access_token":"secret"}}\n'
    auth_file.write_bytes(auth_payload)
    auth_file.chmod(0o600)
    constraints = tmp_path / "workspace/AGENTS.md"
    constraints.parent.mkdir(parents=True)
    constraints_payload = b"# Employee\n\nUse canonical constraints.\n"
    constraints.write_bytes(constraints_payload)
    constraints.chmod(0o600)

    prepare = getattr(env_scope, "prepare_employee_traex_home", None)
    assert prepare is not None
    traex_home = prepare(
        employee_home=str(employee_home),
        auth_file=str(auth_file),
        constraints_file=str(constraints),
    )

    expected_home = employee_home / "runtime/trae-home"
    assert traex_home == str(expected_home)
    assert (expected_home / "cli/auth.json").read_bytes() == auth_payload
    assert (expected_home / "AGENTS.md").read_bytes() == constraints_payload
    assert (expected_home / "cli/auth.json").stat().st_mode & 0o777 == 0o600
    assert (expected_home / "AGENTS.md").stat().st_mode & 0o777 == 0o600

    env = build_employee_process_env(
        {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "manager-key-must-not-leak",
            "LARK_APP_SECRET": "manager-secret-must-not-leak",
        },
        employee_home=str(employee_home),
        credential_env={"TRAE_HOME": traex_home},
    )
    assert env == {
        "HOME": str(employee_home),
        "PATH": "/usr/bin",
        "TRAE_HOME": str(expected_home),
    }


def test_non_traex_backend_never_receives_or_projects_traex_home(tmp_path: Path) -> None:
    employee_home = tmp_path / "employee"
    workspace = employee_home / "workspace"
    workspace.mkdir(parents=True)
    material = EmployeeProcessEnvironmentMaterial(
        tenant_key="tenant_1",
        agent_id="agt_1",
        employee_version=1,
        credential_ref="cred_1",
        runtime_env={"PATH": "/usr/bin"},
        credential_env={"TRAE_HOME": "/shared/manager-trae"},
        provider_files={"traex_auth_json": "/manager/.trae/cli/auth.json"},
    )

    env = build_employee_backend_env(
        material,
        agent_type="coco",
        employee_home=str(employee_home),
        workspace_path=str(workspace),
    )

    assert env == {"HOME": str(employee_home), "PATH": "/usr/bin"}
    assert not (employee_home / "runtime/trae-home").exists()


def test_traex_projection_rejects_group_readable_auth(tmp_path: Path) -> None:
    from src.autonomous.gateway.env_scope import prepare_employee_traex_home

    auth_file = tmp_path / "manager-trae/cli/auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(
        '{"trae":{"access_token":"secret"}}',
        encoding="utf-8",
    )
    auth_file.chmod(0o640)
    constraints = tmp_path / "workspace/AGENTS.md"
    constraints.parent.mkdir(parents=True)
    constraints.write_text("# Employee\n", encoding="utf-8")
    constraints.chmod(0o600)

    with pytest.raises(ValueError, match="auth file is unsafe"):
        prepare_employee_traex_home(
            employee_home=str(tmp_path / "employee"),
            auth_file=str(auth_file),
            constraints_file=str(constraints),
        )

    assert not (tmp_path / "employee").exists()


@pytest.mark.parametrize("backend", ("claude",))
def test_employee_cli_backend_captures_explicit_env_without_switching_to_acp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    env = {"PATH": "/usr/bin", "HOME": str(tmp_path / "employee")}
    monkeypatch.setattr(
        session_factory,
        "get_settings",
        lambda: SimpleNamespace(
            rate_limit_retry_enabled=False,
            acp_startup_timeout=1.0,
        ),
    )
    monkeypatch.setattr(session_factory.SyncClaudeCLISession, "start", lambda self: "sid")

    with employee_session_environment(env):
        session = session_factory.create_engine_session(
            agent_type=backend,
            cwd=str(tmp_path),
            require_tool_filter=True,
        )

    assert "backend=cli" in session.describe_agent()
    assert session.employee_process_env == env
