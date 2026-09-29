"""Codex ACP must resolve its executable from the actual child environment."""

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import src.acp.session as session_mod
import src.acp.transport as transport_mod
from src.acp.session import ACPSession
from src.acp.transport import build_acp_process_env


@pytest.fixture
def codex_environment(tmp_path):
    home = tmp_path / "home"
    bin_dir = home / ".npm-global" / "bin"
    bin_dir.mkdir(parents=True)
    codex = bin_dir / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o755)
    base = {
        "PATH": str(tmp_path / "initial-bin"),
        "HOME": str(home),
        "CODEX_HOME": str(tmp_path / "codex-config"),
        "CLAUDECODE": "nested-session",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    return base, str(codex)


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("npx", ["-y", "@agentclientprotocol/codex-acp@1.2.0"]),
        ("npx", ["@agentclientprotocol/codex-acp"]),
        ("npm", ["exec", "--package=@agentclientprotocol/codex-acp@latest"]),
        ("codex-acp", []),
        ("/opt/homebrew/bin/codex-acp", []),
    ],
)
def test_official_adapter_uses_codex_from_cleaned_child_path(
    command, args, codex_environment
):
    base, codex_path = codex_environment
    original_base = base.copy()
    original_process_env = os.environ.copy()

    env = build_acp_process_env(command, args, base)

    assert env["CODEX_PATH"] == codex_path
    assert codex_path.rsplit(os.sep, 1)[0] in env["PATH"].split(os.pathsep)
    assert "CLAUDECODE" not in env
    for key in ("HOME", "CODEX_HOME", "UV_CACHE_DIR"):
        assert env[key] == base[key]
    assert base == original_base
    assert os.environ == original_process_env


@pytest.mark.parametrize("override", ["", "/operator/custom-codex"])
def test_explicit_codex_path_is_preserved_without_discovery(
    monkeypatch, codex_environment, override
):
    base, _ = codex_environment
    base["CODEX_PATH"] = override
    discover = Mock(side_effect=AssertionError("explicit override must win"))
    monkeypatch.setattr(transport_mod.shutil, "which", discover)

    env = build_acp_process_env("codex-acp", [], base)

    assert env["CODEX_PATH"] == override
    discover.assert_not_called()


def test_missing_installed_codex_keeps_adapter_bundled_fallback(
    monkeypatch, codex_environment
):
    base, _ = codex_environment
    discover = Mock(return_value=None)
    monkeypatch.setattr(transport_mod.shutil, "which", discover)

    env = build_acp_process_env("npx", ["@agentclientprotocol/codex-acp@1.2.0"], base)

    assert "CODEX_PATH" not in env
    discover.assert_called_once_with("codex", path=env["PATH"])
    assert env["PATH"] != base["PATH"]


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("claude", ["acp", "serve"]),
        ("codex", ["exec"]),
        ("npx", ["@zed-industries/codex-acp@0.14.0"]),
        ("npx", ["@other/codex-acp"]),
        ("npx", ["@agentclientprotocol/codex-acp-extra"]),
        ("npx", ["prefix@agentclientprotocol/codex-acp@1.2.0"]),
        ("codex-acp-wrapper", []),
    ],
)
def test_other_commands_do_not_discover_or_inject_codex_path(
    monkeypatch, codex_environment, command, args
):
    base, _ = codex_environment
    discover = Mock(side_effect=AssertionError("unrelated backend must not discover Codex"))
    monkeypatch.setattr(transport_mod.shutil, "which", discover)

    env = build_acp_process_env(command, args, base)

    assert "CODEX_PATH" not in env
    assert "CLAUDECODE" not in env
    discover.assert_not_called()


def test_explicit_base_does_not_inherit_parent_codex_configuration(
    monkeypatch, codex_environment
):
    base, codex_path = codex_environment
    monkeypatch.setenv("CODEX_PATH", "/parent/codex")
    monkeypatch.setenv("CODEX_HOME", "/parent/config")
    monkeypatch.setenv("HOME", "/parent/home")
    original_process_env = os.environ.copy()

    env = build_acp_process_env("codex-acp", [], base)

    assert env["CODEX_PATH"] == codex_path
    assert env["HOME"] == base["HOME"]
    assert env["CODEX_HOME"] == base["CODEX_HOME"]
    assert os.environ == original_process_env


@pytest.mark.parametrize("override", ["", "/parent/custom-codex"])
def test_default_base_preserves_parent_override_without_mutation(monkeypatch, override):
    monkeypatch.setenv("CODEX_PATH", override)
    monkeypatch.setenv("CLAUDECODE", "nested-session")
    original_process_env = os.environ.copy()

    env = build_acp_process_env("codex-acp", [])

    assert env["CODEX_PATH"] == override
    assert "CLAUDECODE" not in env
    assert os.environ == original_process_env


@pytest.mark.asyncio
@pytest.mark.parametrize("use_override", [False, True])
@pytest.mark.parametrize("codex_path_override", [None, "", "/operator/custom-codex"])
async def test_session_start_passes_codex_environment_to_spawn(
    monkeypatch, codex_environment, tmp_path, use_override, codex_path_override
):
    base, installed_codex = codex_environment
    if codex_path_override is not None:
        base["CODEX_PATH"] = codex_path_override
    for key, value in base.items():
        monkeypatch.setenv(key, value)
    if codex_path_override is None:
        monkeypatch.delenv("CODEX_PATH", raising=False)
    original_process_env = os.environ.copy()
    original_base = base.copy()
    conn = SimpleNamespace(
        initialize=AsyncMock(),
        new_session=AsyncMock(return_value=SimpleNamespace(session_id="environment-session")),
        set_config_option=AsyncMock(),
        prompt=AsyncMock(side_effect=AssertionError("startup must not send a prompt")),
    )

    @asynccontextmanager
    async def process_context():
        yield conn, SimpleNamespace(returncode=None)

    spawn = Mock(return_value=process_context())
    monkeypatch.setattr(session_mod, "spawn_agent_process", spawn)
    monkeypatch.setattr(
        session_mod, "get_settings", lambda: SimpleNamespace(acp_stream_buffer_limit=1048576)
    )
    args = ["-y", "@agentclientprotocol/codex-acp@1.2.0"]
    session = ACPSession("npx", args, str(tmp_path), env=base if use_override else None)

    try:
        assert await session.start() == "environment-session"
        spawn.assert_called_once()
        assert spawn.call_args.args[1:] == ("npx", *args)
        child_env = spawn.call_args.kwargs["env"]
        assert child_env["CODEX_PATH"] == (
            installed_codex if codex_path_override is None else codex_path_override
        )
        assert child_env["HOME"] == base["HOME"]
        assert child_env["CODEX_HOME"] == base["CODEX_HOME"]
        assert "CLAUDECODE" not in child_env
        assert spawn.call_args.kwargs["cwd"] == str(tmp_path)
        assert spawn.call_args.kwargs["transport_kwargs"] == {"limit": 1048576}
        conn.initialize.assert_awaited_once_with(protocol_version=1)
        conn.new_session.assert_awaited_once_with(cwd=str(tmp_path))
        conn.set_config_option.assert_awaited_once_with(
            session_id="environment-session", config_id="mode", value="agent-full-access"
        )
        conn.prompt.assert_not_called()
        assert base == original_base
        assert os.environ == original_process_env
    finally:
        await session.close()
