from __future__ import annotations

import types
from pathlib import Path

import pytest

from src.autonomous.workforce.identity import AgentIdentity


def _load_session_host_module():
    import sys

    source_path = Path("src/autonomous/runtime/session_host.py")
    source = source_path.read_text(encoding="utf-8").replace("\x00", "\\x00")
    module = types.ModuleType("src.autonomous.runtime.session_host")
    module.__package__ = "src.autonomous.runtime"
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


class _ProbeSession:
    def __init__(self, result: str = "", *, exception: Exception | None = None) -> None:
        self.result = result
        self.exception = exception
        self.closed = False
        self.cancelled = False
        self.prompts: list[tuple[str, float]] = []
        self.configure_args: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.tool_filter = lambda *args, **kwargs: False  # type: ignore[assignment]

    def configure_employee_sandbox(self, *, read_only_roots, writable_roots) -> None:
        self.configure_args.append((tuple(read_only_roots), tuple(writable_roots)))

    def set_tool_filter(self, tool_filter) -> None:
        self.tool_filter = tool_filter

    def send_prompt(self, prompt: str, *, timeout: float):
        self.prompts.append((prompt, timeout))
        if self.exception:
            raise self.exception
        return self.result

    def is_server_healthy(self) -> bool:
        return not self.closed

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


def _agent(workspace: Path, *, permissions=None, capabilities=None) -> AgentIdentity:
    return AgentIdentity(
        agent_id="agt_session",
        name="Atlas",
        workspace_path=str(workspace),
        security_profile="employee_v1",
        permissions=list(permissions or ["file_read", "file_write", "shell", "git"]),
        capabilities=list(capabilities or ["file_read", "file_write", "shell", "git"]),
    )


def test_session_host_projects_explicit_environment(monkeypatch, tmp_path) -> None:
    from contextlib import contextmanager

    module = _load_session_host_module()
    workspace = tmp_path / "agent" / "workspace"
    workspace.mkdir(parents=True)
    probe = _ProbeSession(result="ok")
    observed_envs: list[dict[str, str] | None] = []
    observed_kwargs: list[dict[str, object]] = []

    def fake_create_engine_session(**kwargs) -> _ProbeSession:
        observed_kwargs.append(kwargs)
        return probe

    @contextmanager
    def fake_employee_session_environment(env):
        observed_envs.append(env)
        yield env

    monkeypatch.setattr(module, "create_engine_session", fake_create_engine_session)
    monkeypatch.setattr(
        module,
        "employee_session_environment",
        fake_employee_session_environment,
    )

    host = module.EmployeeSessionHost()
    env = {"HOME": str(tmp_path / "employee-home"), "PATH": "/usr/bin"}
    lease = host.open_employee_session(_agent(workspace), env=env)
    lease.close()

    assert observed_envs == [env]
    created = observed_kwargs[0]
    assert created["agent_type"] == "coco"
    assert created["cwd"] == str(workspace)
    assert created["thread_id"] == f"employee_actor_{_agent(workspace).agent_id}"
    assert created["auto_approve"] is True
    assert created["require_tool_filter"] is True


def test_run_agent_session_dispatches_one_prompt_with_timeout(monkeypatch, tmp_path) -> None:
    module = _load_session_host_module()
    workspace = tmp_path / "agent" / "workspace"
    workspace.mkdir(parents=True)
    probe = _ProbeSession(result="done")
    create_calls = 0

    def fake_create_engine_session(**_kwargs) -> _ProbeSession:
        nonlocal create_calls
        create_calls += 1
        return probe

    monkeypatch.setattr(module, "create_engine_session", fake_create_engine_session)

    host = module.EmployeeSessionHost()
    result = host.run_agent_session(
        _agent(workspace),
        "execute task",
        timeout=3.5,
        env={"HOME": str(tmp_path / "employee-home"), "PATH": "/usr/bin"},
    )

    assert result == "done"
    assert create_calls == 1
    assert probe.prompts == [("execute task", 3.5)]
    assert probe.closed


def test_run_agent_session_defaults_timeout_and_releases_on_success(monkeypatch, tmp_path) -> None:
    module = _load_session_host_module()
    workspace = tmp_path / "agent" / "workspace"
    workspace.mkdir(parents=True)
    probe = _ProbeSession(result="done")

    monkeypatch.setattr(module, "create_engine_session", lambda **_kwargs: probe)

    host = module.EmployeeSessionHost()
    assert host.run_agent_session(
        _agent(workspace),
        "heartbeat",
        env={"HOME": str(tmp_path / "employee-home"), "PATH": "/usr/bin"},
    ) == "done"
    assert probe.prompts == [("heartbeat", 600.0)]
    assert host._sessions == {}


def test_open_employee_session_tool_filter_blocks_unscoped_paths_and_ungranted_tools(
    monkeypatch, tmp_path
) -> None:
    module = _load_session_host_module()
    workspace = tmp_path / "agent" / "workspace"
    workspace.mkdir(parents=True)
    probe = _ProbeSession(result="ok")

    monkeypatch.setattr(
        module,
        "create_engine_session",
        lambda **_kwargs: probe,
    )

    agent = _agent(
        workspace,
        permissions=["file_read", "git"],
        capabilities=["file_read", "git"],
    )
    host = module.EmployeeSessionHost()
    lease = host.open_employee_session(
        agent,
        env={"HOME": str(tmp_path / "employee-home"), "PATH": "/usr/bin"},
    )
    try:
        tool_filter = probe.tool_filter
        assert tool_filter("file_read", {"path": "notes/README.md"})
        assert not tool_filter("file_read", {"path": str(tmp_path / "outside")})
        assert tool_filter("file_read", {})
        assert not tool_filter("file_write", {"path": "notes/UPD.md"})
        assert not tool_filter("shell", {"command": "pwd"})
        assert not tool_filter("unknown", {})
        assert tool_filter("git", {"path": str(tmp_path / "outside")})
    finally:
        lease.close()

    assert not lease.is_server_healthy()
    assert probe.configure_args == [((str(workspace),), ())]
    assert host._sessions == {}


def test_run_agent_session_fails_without_text_output(monkeypatch, tmp_path) -> None:
    module = _load_session_host_module()
    workspace = tmp_path / "agent" / "workspace"
    workspace.mkdir(parents=True)
    probe = _ProbeSession(result="")

    monkeypatch.setattr(module, "create_engine_session", lambda **_kwargs: probe)

    host = module.EmployeeSessionHost()
    with pytest.raises(
        module.EmployeeSessionUnavailableError,
        match="employee backend returned no output",
    ):
        host.run_agent_session(
            _agent(workspace),
            "echo",
            env={"HOME": str(tmp_path / "employee-home"), "PATH": "/usr/bin"},
        )

    assert probe.closed
    assert host._sessions == {}


def test_run_agent_session_timeout_releases_session(monkeypatch, tmp_path) -> None:
    module = _load_session_host_module()
    workspace = tmp_path / "agent" / "workspace"
    workspace.mkdir(parents=True)
    probe = _ProbeSession(exception=TimeoutError("deadline"))

    monkeypatch.setattr(module, "create_engine_session", lambda **_kwargs: probe)

    host = module.EmployeeSessionHost()
    with pytest.raises(TimeoutError, match="deadline"):
        host.run_agent_session(
            _agent(workspace),
            "long task",
            timeout=0.2,
            env={"HOME": str(tmp_path / "employee-home"), "PATH": "/usr/bin"},
        )

    assert probe.closed
    assert host._sessions == {}
