"""Tests for src.acp.session_factory module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.acp.session_factory import DefaultACPSessionFactory


class TestDefaultACPSessionFactory:
    """Verify routing logic without spawning real sessions."""

    def _make_factory(self):
        settings = MagicMock()
        return DefaultACPSessionFactory(settings)

    @patch("src.agent_session.SyncClaudeCLISession")
    @patch("src.utils.path.normalize_ttadk_cwd", return_value=None)
    def test_claude_type_creates_cli_session(self, _mock_norm, mock_cli):
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

    @patch("src.agent_session.SyncTTADKCLISession")
    @patch("src.utils.path.normalize_ttadk_cwd", return_value=None)
    def test_ttadk_prefix_creates_ttadk_session(self, _mock_norm, mock_ttadk):
        factory = self._make_factory()
        with patch("src.ttadk.startup_common.precheck_ttadk_startup_model", side_effect=RuntimeError("skip")):
            factory.create_session("ttadk_coco", "/tmp")
        mock_ttadk.assert_called_once()
        assert mock_ttadk.call_args.kwargs.get("agent_type") == "ttadk_coco" or \
               mock_ttadk.call_args[1].get("agent_type") == "ttadk_coco"

    @patch("src.acp.sync_adapter.SyncACPSession")
    @patch("src.utils.path.normalize_ttadk_cwd", return_value=None)
    @patch("src.coco_model.get_coco_model_manager")
    def test_coco_type_creates_acp_session(self, mock_mgr, _mock_norm, mock_acp):
        mock_mgr.return_value.get_current_model.return_value = "gpt-4"
        factory = self._make_factory()
        factory.create_session("coco", "/tmp")
        mock_acp.assert_called_once()

    @patch("src.acp.sync_adapter.SyncACPSession")
    @patch("src.utils.path.normalize_ttadk_cwd", return_value=None)
    @patch("src.coco_model.get_coco_model_manager")
    def test_empty_agent_type_defaults_to_coco(self, mock_mgr, _mock_norm, mock_acp):
        mock_mgr.return_value.get_current_model.return_value = None
        factory = self._make_factory()
        factory.create_session("", "/tmp")
        # Should route to ACP with agent_type "coco"
        assert mock_acp.call_args.kwargs.get("agent_type") == "coco" or \
               mock_acp.call_args[1].get("agent_type") == "coco"
