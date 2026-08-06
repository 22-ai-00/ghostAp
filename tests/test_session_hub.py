from unittest.mock import MagicMock

from src.feishu.session_hub import SessionManagerHub


class TestSessionManagerHub:
    def test_initialization(self):
        settings = MagicMock()
        settings.coco_session_timeout = 3600
        settings.claude_session_timeout = 7200
        settings.acp_keepalive_interval = 0  # Use real int to avoid TypeError
        settings.acp_session_idle_healthcheck_s = 60.0

        idle_health_cfg = MagicMock()

        hub = SessionManagerHub(settings, idle_health_cfg)

        # Verify all managers are initialized
        assert hub.coco._agent_type == "coco"
        assert hub.claude._agent_type == "claude"
        assert hub.aiden._agent_type == "aiden"
        assert hub.codex._agent_type == "codex"
        assert hub.gemini._agent_type == "gemini"

        # Verify some settings are passed
        assert hub.coco._session_timeout == 3600
        assert hub.claude._session_timeout == 7200

    def test_cleanup_all(self):
        settings = MagicMock()
        settings.acp_keepalive_interval = 0
        idle_health_cfg = MagicMock()

        hub = SessionManagerHub(settings, idle_health_cfg)

        # Mock cleanup_all for all managers
        hub.coco.cleanup_all = MagicMock()
        hub.claude.cleanup_all = MagicMock()
        hub.aiden.cleanup_all = MagicMock()
        hub.codex.cleanup_all = MagicMock()
        hub.gemini.cleanup_all = MagicMock()

        hub.cleanup_all()

        hub.coco.cleanup_all.assert_called_once()
        hub.claude.cleanup_all.assert_called_once()
        hub.aiden.cleanup_all.assert_called_once()
        hub.codex.cleanup_all.assert_called_once()
        hub.gemini.cleanup_all.assert_called_once()

    def test_cleanup_all_with_exception_isolation(self):
        settings = MagicMock()
        settings.acp_keepalive_interval = 0
        idle_health_cfg = MagicMock()

        hub = SessionManagerHub(settings, idle_health_cfg)

        # Mock cleanup_all to raise exception for one manager
        hub.coco.cleanup_all = MagicMock(side_effect=RuntimeError("cleanup failed"))
        hub.claude.cleanup_all = MagicMock()

        # Should not raise exception
        hub.cleanup_all()

        # Verify other managers are still cleaned up
        hub.claude.cleanup_all.assert_called_once()
