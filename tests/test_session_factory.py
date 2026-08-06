"""Tests for src.acp.session_factory module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.acp.session_factory import DefaultACPSessionFactory


class TestDefaultACPSessionFactory:
    """Verify routing logic without spawning real sessions."""

    def _make_factory(self):
        settings = MagicMock()
        return DefaultACPSessionFactory(settings)

    @patch("src.agent_session.SyncClaudeCLISession")
    def test_claude_type_creates_cli_session(self, mock_cli):
        factory = self._make_factory()
        factory.create_session(
            "claude",
            "/tmp",
            model_name="claude-sonnet-4-5",
        )
        mock_cli.assert_called_once_with(
            cwd="/tmp",
            model_name="claude-sonnet-4-5",
        )


    @patch("src.acp.sync_adapter.SyncACPSession")
    @patch("src.coco_model.get_coco_model_manager")
    def test_coco_type_creates_acp_session(self, mock_mgr, mock_acp):
        mock_mgr.return_value.get_current_model.return_value = "gpt-4"
        factory = self._make_factory()
        factory.create_session("coco", "/tmp")
        mock_acp.assert_called_once()

    @patch("src.acp.sync_adapter.SyncACPSession")
    @patch("src.coco_model.get_coco_model_manager")
    def test_empty_agent_type_defaults_to_coco(self, mock_mgr, mock_acp):
        mock_mgr.return_value.get_current_model.return_value = None
        factory = self._make_factory()
        factory.create_session("", "/tmp")
        # Should route to ACP with agent_type "coco"
        assert mock_acp.call_args.kwargs.get("agent_type") == "coco" or \
               mock_acp.call_args[1].get("agent_type") == "coco"


def test_create_engine_session_normalizes_relative_cwd(monkeypatch, tmp_path: Path):
    """A surviving Coco session receives a stable absolute project path."""
    import src.acp.sync_adapter as sync_adapter
    from src import agent_session

    monkeypatch.chdir(tmp_path)
    captured: dict[str, str] = {}

    def fake_start_session_with_retry(*, cwd: str, **_kwargs):
        captured["cwd"] = cwd

        class FakeSession:
            def start(self, startup_timeout=None):
                return None

            def close(self):
                return None

        return FakeSession()

    monkeypatch.setattr(
        sync_adapter,
        "start_session_with_retry",
        fake_start_session_with_retry,
    )

    agent_session.create_engine_session(agent_type="coco", cwd=".")

    assert Path(captured["cwd"]) == tmp_path.resolve()
