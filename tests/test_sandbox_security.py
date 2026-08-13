"""Tests for sandbox security hardening (Task A1).

Covers:
- DangerousPatternCheckStrategy destructive-command blocking
- redact_sensitive integration in _execute_command
- set_working_dir path range validation
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox.executor import DangerousPatternCheckStrategy, SandboxExecutor

# ---------------------------------------------------------------------------
# DangerousPatternCheckStrategy: precise destructive-command blocking
# ---------------------------------------------------------------------------

class TestDangerousPatterns:
    """Normal shell composition is allowed; destructive patterns are not."""

    @pytest.fixture
    def strategy(self):
        return DangerousPatternCheckStrategy()

    @pytest.fixture
    def settings(self):
        """Minimal settings mock (fields are not needed for DangerousPattern)."""
        return MagicMock()

    def test_allows_safe_shell_composition(self, strategy, settings):
        command = (
            "zsh -c \"cd /workspace && python - <<'PY' > /tmp/result\n"
            "print('ok')\nPY\ncat /tmp/result | sed -n '1p'\""
        )

        is_safe, reason = strategy.check(command, settings)

        assert is_safe is True
        assert reason is None

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "zsh -c 'rm -rf /'",
            "rm -rf /; echo should-not-run",
            "rm -rf / && echo should-not-run",
        ],
    )
    def test_blocks_destructive_root_removal(self, strategy, settings, command):
        is_safe, reason = strategy.check(command, settings)

        assert is_safe is False
        assert "危险操作模式" in reason

    def test_allows_safe_command(self, strategy, settings):
        is_safe, reason = strategy.check("ls -la /tmp", settings)
        assert is_safe is True
        assert reason is None


def test_sandbox_keeps_configured_blacklist_and_whitelist_gates() -> None:
    blocked_by_blacklist = SandboxExecutor(
        settings=SimpleNamespace(
            sandbox_use_whitelist=False,
            command_whitelist=[],
            command_blacklist=["secret-tool"],
        )
    )
    blocked_by_whitelist = SandboxExecutor(
        settings=SimpleNamespace(
            sandbox_use_whitelist=True,
            command_whitelist=["git"],
            command_blacklist=[],
        )
    )

    assert blocked_by_blacklist.is_command_safe("secret-tool run")[0] is False
    assert blocked_by_whitelist.is_command_safe("python build.py")[0] is False


# ---------------------------------------------------------------------------
# redact_sensitive called on executor output
# ---------------------------------------------------------------------------

class TestRedactSensitiveInExecutor:
    """SandboxExecutor._execute_command must call redact_sensitive on stdout/stderr."""

    def test_redact_called_on_output(self):
        mock_settings = MagicMock()
        mock_settings.sandbox_timeout = 10
        mock_settings.sandbox_max_output_length = 4000
        mock_settings.sandbox_use_whitelist = False
        mock_settings.command_blacklist = []
        mock_settings.command_whitelist = []

        # Build a mock subprocess executor that returns sensitive data
        mock_subprocess = MagicMock()
        mock_process = MagicMock()
        mock_process.stdout = "token: sk-abcdefghijklmnop1234"
        mock_process.stderr = "secret= mysecretvalue123"
        mock_process.returncode = 0
        mock_subprocess.run.return_value = mock_process

        executor = SandboxExecutor(
            settings=mock_settings,
            subprocess_executor=mock_subprocess,
            security_strategies=[],  # skip security checks for this test
        )

        with patch("src.sandbox.executor.redact_sensitive", wraps=__import__("src.utils.redact", fromlist=["redact_sensitive"]).redact_sensitive) as mock_redact:
            result = executor._execute_command("echo test", cwd="/tmp", interactive=False)
            assert mock_redact.call_count >= 2  # called for stdout and stderr

        # Verify sensitive data was actually redacted in output
        assert "sk-abcdefghijklmnop1234" not in result.stdout
        assert "mysecretvalue123" not in result.stderr
        assert "<redacted>" in result.stdout
        assert "<redacted>" in result.stderr


# ---------------------------------------------------------------------------
# set_working_dir path range validation
# ---------------------------------------------------------------------------

class TestSetWorkingDirPathValidation:
    """BaseHandler.set_working_dir must reject paths outside project_allowed_roots."""

    def _make_handler(self):
        """Create a minimal BaseHandler-like object with set_working_dir."""
        from src.feishu.handlers.base import BaseHandler

        # BaseHandler requires ctx; mock just enough
        mock_ctx = MagicMock()
        mock_ctx.working_dirs = {}
        mock_ctx.working_dir_lock = __import__("threading").Lock()

        handler = object.__new__(BaseHandler)
        handler.ctx = mock_ctx
        return handler

    def test_rejects_path_outside_allowed_roots(self, tmp_path):
        handler = self._make_handler()

        # Use a temp path that is NOT under ~/workspaces
        outside_dir = str(tmp_path / "evil")
        os.makedirs(outside_dir, exist_ok=True)

        handler.ctx.settings.project_allowed_roots = ["/Users/allowed/workspaces"]
        ok, msg = handler.set_working_dir("chat123", outside_dir)
        assert ok is False
        assert "不在允许范围内" in msg

    def test_accepts_path_within_allowed_roots(self, tmp_path):
        handler = self._make_handler()

        # Create a dir inside an allowed root
        allowed_root = str(tmp_path / "workspaces")
        project_dir = os.path.join(allowed_root, "myproject")
        os.makedirs(project_dir, exist_ok=True)

        handler.ctx.settings.project_allowed_roots = [allowed_root]
        ok, msg = handler.set_working_dir("chat123", project_dir)
        assert ok is True
        assert msg == project_dir

    def test_allows_any_path_when_roots_empty(self, tmp_path):
        handler = self._make_handler()

        target_dir = str(tmp_path)

        handler.ctx.settings.project_allowed_roots = []
        ok, msg = handler.set_working_dir("chat123", target_dir)
        assert ok is True
