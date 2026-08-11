"""Tests for get_providers() / _ensure_providers() single-lock thread-safety."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import src.acp.providers as providers_mod
from src.acp.providers import GenericACPProvider, get_providers


def _reset_providers_cache() -> None:
    """Reset module-level caches so each test starts fresh."""
    providers_mod._providers = None
    providers_mod._checkers = None


def _fake_checker_tuple() -> tuple:
    def noop():
        return True
    def noop_loader():
        return ""
    def noop_clear():
        return None
    return (noop, noop_loader, noop_clear)


class TestGetProvidersThreadSafety:
    """Verify that get_providers() uses proper double-checked locking with a single lock."""

    def setup_method(self) -> None:
        _reset_providers_cache()

    def teardown_method(self) -> None:
        _reset_providers_cache()

    def test_concurrent_first_call_builds_only_once(self) -> None:
        """Multiple threads hitting get_providers() concurrently should
        trigger the expensive build path exactly once."""
        build_count = 0
        build_lock = threading.Lock()
        barrier = threading.Barrier(8, timeout=5)

        fake_tuple = _fake_checker_tuple()

        def _counting_custom(*a, **kw):
            nonlocal build_count
            with build_lock:
                build_count += 1
            return fake_tuple

        with patch.object(providers_mod, "_make_custom_help_checker_with_cache_handle",
                          side_effect=_counting_custom), \
             patch.object(providers_mod, "_make_probe_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_resolve_checker", return_value=lambda: True):

            def _worker() -> dict[str, GenericACPProvider]:
                barrier.wait()
                return get_providers()

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_worker) for _ in range(8)]
                results = [f.result(timeout=10) for f in as_completed(futures)]

        # _make_custom_help_checker_with_cache_handle is called four times
        # (aiden + gemini + traex + grok) but only in ONE init path.
        assert build_count == 4, f"Expected 4 checker builds (aiden+gemini+traex+grok), got {build_count}"

        # All threads must have received the same dict object (identity check)
        first = results[0]
        assert all(r is first for r in results), "All threads should get the same cached object"

    def test_fast_path_after_cache_populated(self) -> None:
        """Once the cache is populated, subsequent calls bypass the lock."""
        fake_tuple = _fake_checker_tuple()

        with patch.object(providers_mod, "_make_custom_help_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_probe_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_resolve_checker", return_value=lambda: True):
            first = get_providers()

        # Now cache is populated — calling again should hit fast path only.
        # Patch builder to raise; if it's called, the fast path is broken.
        with patch.object(providers_mod, "_make_custom_help_checker_with_cache_handle",
                          side_effect=AssertionError("should not rebuild")):
            second = get_providers()

        assert second is first


class TestProvidersCacheInvalidation:
    """Verify cache can be invalidated and correctly rebuilt."""

    def setup_method(self) -> None:
        _reset_providers_cache()

    def teardown_method(self) -> None:
        _reset_providers_cache()

    def test_reset_cache_triggers_rebuild(self) -> None:
        """After manually clearing _providers, next call rebuilds."""
        fake_tuple = _fake_checker_tuple()

        with patch.object(providers_mod, "_make_custom_help_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_probe_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_resolve_checker", return_value=lambda: True):
            first = get_providers()

        # Manually invalidate
        _reset_providers_cache()

        with patch.object(providers_mod, "_make_custom_help_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_probe_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_resolve_checker", return_value=lambda: True):
            second = get_providers()

        # Should be a new dict (different object identity) but same keys
        assert second is not first
        assert set(second.keys()) == set(first.keys())

    def test_rebuild_produces_valid_providers(self) -> None:
        """Rebuilt providers should have all expected tool names."""
        fake_tuple = _fake_checker_tuple()

        with patch.object(providers_mod, "_make_custom_help_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_probe_checker_with_cache_handle",
                          return_value=fake_tuple), \
             patch.object(providers_mod, "_make_resolve_checker", return_value=lambda: True):
            providers = get_providers()

        expected_names = {"coco", "claude", "aiden", "codex", "gemini", "traex", "grok"}
        assert set(providers.keys()) == expected_names

    def test_single_lock_no_intermediate_caches(self) -> None:
        """Verify the module only has one lock (_init_lock), no intermediate cache globals."""
        # The old _checker_cache_lock, _provider_configs_lock, _providers_lock should be gone
        assert not hasattr(providers_mod, "_checker_cache_lock")
        assert not hasattr(providers_mod, "_provider_configs_lock")
        assert not hasattr(providers_mod, "_providers_lock")
        assert not hasattr(providers_mod, "_checker_cache")
        assert not hasattr(providers_mod, "_PROVIDER_CONFIGS_CACHE")
        assert hasattr(providers_mod, "_init_lock")
