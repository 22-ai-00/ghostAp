from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent_session.employee_cli_sandbox import (
    EmployeeCLISandbox,
    EmployeeCLISandboxError,
)
from src.autonomous.runtime.session_host import EmployeeSessionHost
from src.autonomous.workforce.identity import AgentIdentity


def test_employee_session_host_limits_tools_to_employee_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    workspace = tmp_path / "agents/agt_1/workspace"
    project.mkdir()
    workspace.mkdir(parents=True)
    session = MagicMock()
    agent = AgentIdentity(
        agent_id="agt_1",
        name="Atlas",
        workspace_path=str(workspace),
        security_profile="employee_v1",
        permissions=["file_read", "file_write", "shell", "git"],
        capabilities=["file_read", "file_write", "shell", "git"],
    )

    monkeypatch.setattr(
        "src.autonomous.runtime.session_host.create_engine_session",
        lambda **_kwargs: session,
    )
    host = EmployeeSessionHost()
    lease = host.open_employee_session(agent, env={"PATH": "/usr/bin"})
    tool_filter = session.set_tool_filter.call_args.args[0]

    assert tool_filter("file_read", {"path": str(workspace / "IDENTITY.md")}) is True
    assert tool_filter("file_write", {"path": str(workspace / "NOW.md")}) is True
    assert tool_filter("file_write", {"path": str(project / "result.txt")}) is False
    assert tool_filter("file_read", {"path": str(project / ".env")}) is False
    assert tool_filter("file_read", {"path": str(tmp_path / "vault/key")}) is False
    assert tool_filter("shell", {"command": "pwd", "cwd": str(workspace)}) is True
    assert tool_filter("shell", {"command": "pwd", "cwd": str(project)}) is True
    session.configure_employee_sandbox.assert_called_once_with(
        read_only_roots=(str(workspace),),
        writable_roots=(str(workspace),),
    )
    lease.close()
    host.close()


def test_employee_cli_namespace_hides_sensitive_and_peer_employee_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    employee = tmp_path / "agents/agt_1"
    workspace = employee / "workspace"
    peer = tmp_path / "agents/agt_2"
    project.mkdir()
    workspace.mkdir(parents=True)
    peer.mkdir(parents=True)
    (project / ".env").write_text("SECRET=never-visible", encoding="utf-8")
    (project / "vault").mkdir()
    (project / "journal").mkdir()
    sandbox = EmployeeCLISandbox(
        cwd=str(project),
        process_env={"PATH": "/usr/bin", "HOME": str(employee)},
    )

    sandbox.configure(
        command="sh",
        read_only_roots=(str(project), str(workspace)),
        writable_roots=(str(project),),
    )
    output = project / "output.txt"
    argv = sandbox.wrap_argv(
        ["sh", "-c", f"test ! -s {project / '.env'} && printf ok > {output}"]
    )
    rendered = "\0".join(argv)

    assert argv[0].endswith("bwrap")
    assert f"--bind\0{project}\0{project}" in rendered
    assert f"--ro-bind\0{workspace}\0{workspace}" in rendered
    assert f"--ro-bind\0/dev/null\0{project / '.env'}" in rendered
    assert f"--tmpfs\0{project / 'vault'}" in rendered
    assert f"--tmpfs\0{project / 'journal'}" in rendered
    assert str(peer) not in rendered
    assert subprocess.run(argv, check=False).returncode == 0
    assert output.read_text(encoding="utf-8") == "ok"


def test_employee_cli_refuses_spawn_before_filesystem_policy() -> None:
    sandbox = EmployeeCLISandbox(
        cwd="/tmp",
        process_env={"PATH": "/usr/bin", "HOME": "/tmp/employee"},
    )

    with pytest.raises(EmployeeCLISandboxError, match="not configured"):
        sandbox.wrap_argv(["true"])
