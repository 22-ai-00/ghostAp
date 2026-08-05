from __future__ import annotations


def test_workflow_model_lookup_uses_short_ttl_cache(monkeypatch):
    from src.feishu.handlers.workflow import WorkflowHandler

    calls: list[tuple[str, str, str, bool]] = []

    class FakeDiscovery:
        def get_models_for_tool(
            self,
            tool_name,
            provider="ttadk",
            cwd=None,
            current_model=None,
            force_refresh=True,
        ):
            calls.append((tool_name, provider, cwd or "", force_refresh))
            return [{"name": "openrouter-3o/low", "display_name": "openrouter-3o/low"}]

    monkeypatch.setattr(
        "src.acp.tool_discovery.AgentToolDiscovery",
        FakeDiscovery,
    )

    handler = WorkflowHandler.__new__(WorkflowHandler)
    first = handler._get_workflow_models_for_tool("traex", "/repo")
    second = handler._get_workflow_models_for_tool("traex", "/repo")

    assert first == second
    assert calls == [("traex", "acp", "/repo", False)]


def test_workflow_model_lookup_does_not_rescan_all_tools(monkeypatch):
    from src.feishu.handlers.workflow import WorkflowHandler

    scans: list[str] = []
    calls: list[tuple[str, str, str, bool]] = []

    class FakeDiscovery:
        def get_available_tools(self):
            scans.append("called")
            return [{"tool_name": "traex", "provider": "ttadk"}]

        def get_models_for_tool(
            self,
            tool_name,
            provider="ttadk",
            cwd=None,
            current_model=None,
            force_refresh=True,
        ):
            calls.append((tool_name, provider, cwd or "", force_refresh))
            return [{"name": "openrouter-3o/low", "display_name": "openrouter-3o/low"}]

    monkeypatch.setattr(
        "src.acp.tool_discovery.AgentToolDiscovery",
        FakeDiscovery,
    )

    handler = WorkflowHandler.__new__(WorkflowHandler)
    models = handler._get_workflow_models_for_tool("traex", "/repo")

    assert models == [{"name": "openrouter-3o/low", "display_name": "openrouter-3o/low", "description": ""}]
    assert scans == []
    assert calls == [("traex", "acp", "/repo", False)]


def test_workflow_ttdak_model_selection_does_not_force_refresh(monkeypatch):
    from src.feishu.handlers.workflow import WorkflowHandler

    calls: list[tuple[str, str, str, bool]] = []

    class FakeDiscovery:
        def get_models_for_tool(
            self,
            tool_name,
            provider="ttadk",
            cwd=None,
            current_model=None,
            force_refresh=True,
        ):
            calls.append((tool_name, provider, cwd or "", force_refresh))
            return [{"name": "doubao-seed", "display_name": "doubao-seed"}]

    monkeypatch.setattr(
        "src.acp.tool_discovery.AgentToolDiscovery",
        FakeDiscovery,
    )

    handler = WorkflowHandler.__new__(WorkflowHandler)
    models = handler._get_workflow_models_for_tool(
        "doubao",
        "/repo",
    )

    assert calls == [("doubao", "ttadk", "/repo", False)]
    assert models == [{"name": "doubao-seed", "display_name": "doubao-seed", "description": ""}]


def test_workflow_codex_model_lookup_preserves_effort_capabilities_and_cache(
    monkeypatch,
):
    from src.feishu.handlers.workflow import WorkflowHandler

    calls = 0

    class FakeDiscovery:
        def get_models_for_tool(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return [
                {
                    "name": "gpt-5.6-sol",
                    "display_name": "GPT-5.6-Sol",
                    "description": "Latest frontier model",
                    "is_default": True,
                    "reasoning_efforts": ["low", "high", "max", "ultra"],
                    "adapted_reasoning_effort": "high",
                }
            ]

    monkeypatch.setattr(
        "src.acp.tool_discovery.AgentToolDiscovery",
        FakeDiscovery,
    )

    handler = WorkflowHandler.__new__(WorkflowHandler)
    first = handler._get_workflow_models_for_tool("codex", "/repo")
    first[0]["reasoning_efforts"].append("mutated")
    second = handler._get_workflow_models_for_tool("codex", "/repo")

    assert calls == 1
    assert second == [
        {
            "name": "gpt-5.6-sol",
            "display_name": "GPT-5.6-Sol",
            "description": "Latest frontier model",
            "is_default": True,
            "reasoning_efforts": ["low", "high", "max", "ultra"],
            "adapted_reasoning_effort": "high",
        }
    ]


def test_workflow_model_lookup_preserves_explicit_variants_and_copy_isolation(
    monkeypatch,
):
    from src.feishu.handlers.workflow import WorkflowHandler

    calls = 0

    class FakeDiscovery:
        def get_models_for_tool(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return [
                {
                    "name": "c_o_new_thinking",
                    "display_name": "Test-O-New-Thinking",
                    "selection_variants": [
                        {
                            "name": "c_o_new_thinking/max/max",
                            "profile": "max",
                            "effort": "max",
                        }
                    ],
                }
            ]

    monkeypatch.setattr(
        "src.acp.tool_discovery.AgentToolDiscovery",
        FakeDiscovery,
    )
    handler = WorkflowHandler.__new__(WorkflowHandler)

    first = handler._get_workflow_models_for_tool("traex", "/repo")
    first[0]["selection_variants"][0]["effort"] = "mutated"
    second = handler._get_workflow_models_for_tool("traex", "/repo")

    assert calls == 1
    assert second[0]["selection_variants"][0]["effort"] == "max"


def test_workflow_model_cache_reloads_after_shared_generation_changes(monkeypatch):
    from src.acp.helper import invalidate_acp_model_cache
    from src.feishu.handlers.workflow import WorkflowHandler

    calls = 0

    class FakeDiscovery:
        def get_models_for_tool(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return [{"name": f"traex-model-{calls}"}]

    monkeypatch.setattr(
        "src.acp.tool_discovery.AgentToolDiscovery",
        FakeDiscovery,
    )
    handler = WorkflowHandler.__new__(WorkflowHandler)

    first = handler._get_workflow_models_for_tool("traex", "/repo")
    invalidate_acp_model_cache("traex", "/repo")
    second = handler._get_workflow_models_for_tool("traex", "/repo")

    assert first[0]["name"] == "traex-model-1"
    assert second[0]["name"] == "traex-model-2"
    assert calls == 2
