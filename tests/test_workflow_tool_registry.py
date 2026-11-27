"""Tests for src/workflow_engine/tool_registry.py — dynamic tool discovery."""

import unittest
from unittest.mock import patch

from src.workflow_engine.tool_registry import (
    _FALLBACK_DESCRIPTIONS,
    _discover_acp_tools,
    get_available_tools,
    invalidate_cache,
)


class TestGetAvailableTools(unittest.TestCase):
    """Test get_available_tools returns a valid tool dict."""

    def setUp(self):
        invalidate_cache()

    def tearDown(self):
        invalidate_cache()

    def test_contains_coco(self):
        tools = get_available_tools()
        self.assertIn("coco", tools)

    def test_all_values_are_strings(self):
        tools = get_available_tools()
        for name, desc in tools.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(desc, str)
            self.assertTrue(len(desc) > 0, f"{name} has empty description")

    def test_fallback_on_import_error(self):
        """When ACP imports fail, returns fallback descriptions."""
        with patch(
            "src.workflow_engine.tool_registry._discover_acp_tools",
            side_effect=Exception("no acp"),
        ):
            invalidate_cache()
            tools = get_available_tools(force_refresh=True)
            # Should still have all fallback tools
            for name in _FALLBACK_DESCRIPTIONS:
                self.assertIn(name, tools)

    def test_cache_returns_copy(self):
        """Returned dict is a copy — mutations don't affect cache."""
        tools1 = get_available_tools()
        tools1["MUTATED"] = "should not persist"
        tools2 = get_available_tools()
        self.assertNotIn("MUTATED", tools2)

    def test_force_refresh(self):
        """force_refresh=True re-discovers even with warm cache."""
        call_count = [0]

        def counting_discover(**_kwargs):
            call_count[0] += 1
            return dict(_FALLBACK_DESCRIPTIONS)

        with patch(
            "src.workflow_engine.tool_registry._discover_tools",
            side_effect=counting_discover,
        ):
            invalidate_cache()
            get_available_tools(force_refresh=True)
            get_available_tools(force_refresh=True)
            self.assertEqual(call_count[0], 2)

    def test_require_available_does_not_add_fallback_tools(self):
        """Selectable UI mode must not show tools that were only fallback names."""
        with patch(
            "src.workflow_engine.tool_registry._discover_acp_tools",
            return_value={},
        ):
            invalidate_cache()
            tools = get_available_tools(force_refresh=True, require_available=True)
            self.assertEqual(tools, {})

    def test_discover_acp_tools_filters_unavailable_providers(self):
        """require_available=True keeps only providers passing cached probes when no binary exists."""
        with patch(
            "src.acp.providers.get_providers",
            return_value={"coco": object(), "codex": object()},
        ), patch(
            "shutil.which",
            return_value=None,
        ), patch(
            "src.acp.providers.tool_registry.get_availability",
            side_effect=lambda name, **_: name == "coco",
        ), patch(
            "src.utils.text.get_acp_result_header_text",
            return_value={"tool_desc_coco": "Coco OK", "tool_desc_codex": "Codex NO"},
        ):
            tools = _discover_acp_tools(require_available=True)

        self.assertEqual(tools, {"coco": "Coco OK"})

    def test_discover_acp_tools_uses_fast_binary_check_without_sync_probe(self):
        """Selectable tool discovery must not block on ACP probes just to build /wf cards."""
        seen: list[dict] = []

        def _availability(_name, **kwargs):
            seen.append(dict(kwargs))
            if kwargs.get("allow_sync_probe"):
                raise AssertionError("selectable tool discovery used a synchronous ACP probe")
            return False

        with patch(
            "src.acp.providers.get_providers",
            return_value={"traex": object()},
        ), patch(
            "shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}" if name == "traex" else None,
        ), patch(
            "src.acp.providers.tool_registry.get_availability",
            side_effect=_availability,
        ), patch(
            "src.utils.text.get_acp_result_header_text",
            return_value={"tool_desc_traex": "Traex OK"},
        ):
            tools = _discover_acp_tools(require_available=True)

        self.assertEqual(tools, {"traex": "Traex OK"})
        self.assertTrue(all(not call.get("allow_sync_probe") for call in seen))

    def test_claude_cli_binary_does_not_trigger_acp_warmup(self):
        """Claude uses its CLI transport; only ACP tools receive ACP warmup."""
        availability = unittest.mock.MagicMock(return_value=True)

        with patch(
            "src.acp.providers.get_providers",
            return_value={"claude": object(), "codex": object()},
        ), patch(
            "shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ), patch(
            "src.acp.providers.tool_registry.get_availability",
            side_effect=availability,
        ), patch(
            "src.utils.text.get_acp_result_header_text",
            return_value={
                "tool_desc_claude": "Claude CLI",
                "tool_desc_codex": "Codex ACP",
            },
        ):
            tools = _discover_acp_tools(require_available=True)

        self.assertEqual(
            tools,
            {"claude": "Claude CLI", "codex": "Codex ACP"},
        )
        self.assertEqual(
            [call.args[0] for call in availability.call_args_list],
            ["codex"],
        )

    def test_missing_claude_cli_does_not_fall_back_to_acp_probe(self):
        """A missing Claude CLI is unavailable rather than an ACP candidate."""
        availability = unittest.mock.MagicMock(return_value=True)

        with patch(
            "src.acp.providers.get_providers",
            return_value={"claude": object(), "codex": object()},
        ), patch(
            "shutil.which",
            side_effect=lambda name: "/usr/bin/codex" if name == "codex" else None,
        ), patch(
            "src.acp.providers.tool_registry.get_availability",
            side_effect=availability,
        ), patch(
            "src.utils.text.get_acp_result_header_text",
            return_value={
                "tool_desc_claude": "Claude CLI",
                "tool_desc_codex": "Codex ACP",
            },
        ):
            tools = _discover_acp_tools(require_available=True)

        self.assertEqual(tools, {"codex": "Codex ACP"})
        self.assertEqual(
            [call.args[0] for call in availability.call_args_list],
            ["codex"],
        )

    def test_discover_acp_tools_detects_agy_b_via_hyphenated_binary_without_sync_probe(self):
        """Cold-cache discovery must probe shutil.which('agy-b'), not 'agy_b', without sync probes."""
        from types import SimpleNamespace

        seen_which: list[str] = []
        seen_availability: list[dict] = []

        def _which(cmd: str):
            seen_which.append(cmd)
            return "/usr/local/bin/agy-b" if cmd == "agy-b" else None

        def _availability(_name, **kwargs):
            seen_availability.append(dict(kwargs))
            if kwargs.get("allow_sync_probe"):
                raise AssertionError("cold-cache discovery must not run synchronous ACP probe")
            return False

        # Test both _config.binary_name and backend_spec.cli_name fallback (plain object)
        for provider_obj in (
            SimpleNamespace(_config=SimpleNamespace(binary_name="agy-b")),
            object(),
        ):
            seen_which.clear()
            seen_availability.clear()
            with patch(
                "src.acp.providers.get_providers",
                return_value={"agy_b": provider_obj},
            ), patch(
                "shutil.which",
                side_effect=_which,
            ), patch(
                "src.acp.providers.tool_registry.get_availability",
                side_effect=_availability,
            ):
                tools = _discover_acp_tools(require_available=True)

            self.assertEqual(
                tools,
                {"agy_b": "Google Antigravity 原生 ACP 编程 (AGY-B)"},
            )
            self.assertIn("agy-b", seen_which)
            self.assertTrue(all(not call.get("allow_sync_probe") for call in seen_availability))

    def test_discover_acp_tools_agy_b_uses_fallback_description_when_text_missing(self):
        """When text.get_acp_result_header_text() lacks tool_desc_agy_b, fallback description is used."""
        with patch(
            "src.acp.providers.get_providers",
            return_value={"agy_b": object()},
        ), patch(
            "shutil.which",
            side_effect=lambda cmd: "/usr/local/bin/agy-b" if cmd == "agy-b" else None,
        ), patch(
            "src.acp.providers.tool_registry.get_availability",
            return_value=False,
        ), patch(
            "src.utils.text.get_acp_result_header_text",
            return_value={},
        ):
            tools = _discover_acp_tools(require_available=True)

        self.assertEqual(
            tools,
            {"agy_b": _FALLBACK_DESCRIPTIONS["agy_b"]},
        )
        self.assertEqual(
            _FALLBACK_DESCRIPTIONS["agy_b"],
            "Google Antigravity 原生 ACP 编程 (AGY-B)",
        )


class TestInvalidateCache(unittest.TestCase):
    """Test cache invalidation."""

    def test_invalidate_forces_rediscovery(self):
        # Warm cache
        get_available_tools()
        invalidate_cache()

        call_count = [0]

        def counting_discover(**_kwargs):
            call_count[0] += 1
            return dict(_FALLBACK_DESCRIPTIONS)

        with patch(
            "src.workflow_engine.tool_registry._discover_tools",
            side_effect=counting_discover,
        ):
            get_available_tools()
            self.assertEqual(call_count[0], 1)


if __name__ == "__main__":
    unittest.main()
