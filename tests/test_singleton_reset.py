"""Verify that _reset_*_for_testing() functions properly reset global singletons."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCocoModelManagerReset:
    def test_reset_produces_new_instance(self):
        from src.coco_model.manager import (
            _reset_coco_model_manager_for_testing,
            get_coco_model_manager,
        )

        with patch("src.coco_model.manager.CocoModelManager") as MockCls:
            MockCls.return_value = MagicMock(name="mgr_a")
            a = get_coco_model_manager()
            assert get_coco_model_manager() is a

            _reset_coco_model_manager_for_testing()

            MockCls.return_value = MagicMock(name="mgr_b")
            b = get_coco_model_manager()
            assert b is not a


class TestChatLockManagerSingletonInterface:
    def test_set_chat_lock_manager_injects_test_instance_and_reset_clears_it(self):
        from src.chat_lock import (
            _reset_chat_lock_manager_for_testing,
            get_chat_lock_manager,
            set_chat_lock_manager,
        )

        injected = MagicMock(name="chat_lock_manager")

        set_chat_lock_manager(injected)
        assert get_chat_lock_manager() is injected

        _reset_chat_lock_manager_for_testing()

        with patch("src.chat_lock.ChatLockManager") as MockCls:
            MockCls.return_value = MagicMock(name="rebuilt_chat_lock_manager")
            rebuilt = get_chat_lock_manager()

        assert rebuilt is not injected

    def test_set_and_reset_chat_lock_manager_are_test_only(self):
        from src.chat_lock import _reset_chat_lock_manager_for_testing, set_chat_lock_manager

        injected = MagicMock(name="chat_lock_manager")

        try:
            with patch("src.chat_lock.is_test_environment", return_value=False):
                try:
                    set_chat_lock_manager(injected)
                except RuntimeError as exc:
                    assert "only allowed in test environments" in str(exc)
                else:
                    raise AssertionError("set_chat_lock_manager should reject production injection")

                try:
                    _reset_chat_lock_manager_for_testing()
                except RuntimeError as exc:
                    assert "only allowed in test environments" in str(exc)
                else:
                    raise AssertionError("_reset_chat_lock_manager_for_testing should reject production reset")
        finally:
            _reset_chat_lock_manager_for_testing()


class TestProvidersReset:
    def test_reset_clears_and_allows_rebuild(self):
        from src.acp.providers import (
            _reset_providers_for_testing,
            get_providers,
        )

        # Force a build (may be a no-op if already built by another test)
        with patch("src.acp.providers._make_resolve_checker", return_value=lambda: True), \
             patch("src.acp.providers._make_custom_help_checker_with_cache_handle",
                   return_value=(lambda: True, lambda: "blob", lambda: None)), \
             patch("src.acp.providers._make_probe_checker_with_cache_handle",
                   return_value=(lambda: True, lambda: "blob", lambda: None)):
            providers_a = get_providers()
            assert providers_a is not None

        _reset_providers_for_testing()

        # After reset, the module-level _providers should be None
        from src.acp.providers import _providers as _post_check
        assert _post_check is None


class TestToolRegistryResetForTesting:
    """Verify ToolRegistry._reset_for_testing acquires _lock."""

    def test_reset_removes_providers_under_lock(self):
        from src.acp.provider import ToolRegistry

        reg = ToolRegistry()

        # Build a minimal provider stub
        class _Stub:
            name = "alpha"
            skip_model_selection = False
            def get_serve_command(self, m=None): return ("alpha", [])
            def check_availability(self): return True
            def get_fallback_command(self, m=None): return None

        reg.register(_Stub(), is_default=True)
        assert reg.get_provider("alpha") is not None
        assert reg._default_provider == "alpha"

        reg._reset_for_testing(["alpha"])

        assert reg.get_provider("alpha") is None
        assert reg._default_provider is None

    def test_reset_acquires_lock(self):
        """Confirm _reset_for_testing uses self._lock (via mock)."""
        from unittest.mock import MagicMock

        from src.acp.provider import ToolRegistry

        reg = ToolRegistry()
        real_lock = reg._lock
        mock_lock = MagicMock(wraps=real_lock)
        reg._lock = mock_lock

        reg._reset_for_testing([])

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
