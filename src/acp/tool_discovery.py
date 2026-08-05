from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

from ..ttadk import get_ttadk_manager
from .helper import fetch_acp_models
from .providers import get_providers, tool_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentToolOption:
    provider: str
    tool_name: str
    display_name: str
    agent_name: str = ""
    description: str = ""
    supports_model: bool = True
    model_optional: bool = False
    skip_model_selection: bool = False


@dataclass(frozen=True)
class _KnownTool:
    """Static definition of a top-level tool candidate."""

    name: str
    display_name: str
    description: str
    priority: int


_KNOWN_TOOLS: tuple[_KnownTool, ...] = (
    _KnownTool("coco", "Coco", "字节跳动 AI", 0),
    _KnownTool("aiden", "Aiden", "Aiden CLI", 1),
    _KnownTool("codex", "Codex", "OpenAI Codex", 2),
    _KnownTool("claude", "Claude", "Anthropic Claude CLI", 3),
    _KnownTool("traex", "Traex", "TRAE CLI", 4),
)


class AgentToolDiscovery:
    """Discover available tools and their models from ACP, CLI, and TTADK."""

    _TOP_LEVEL_PRIORITY = {
        ("acp", "coco"): 0,
        ("cli", "coco"): 0,
        ("acp", "aiden"): 1,
        ("cli", "aiden"): 1,
        ("acp", "codex"): 2,
        ("cli", "codex"): 2,
        ("acp", "claude"): 3,
        ("cli", "claude"): 3,
        ("acp", "traex"): 4,
        ("cli", "traex"): 4,
        ("ttadk", "ttadk"): 90,
    }

    def get_available_tools(self) -> list[dict]:
        """Return available tools as dictionaries suitable for selection cards."""
        get_providers()
        tools: list[dict] = []
        seen: set[str] = set()

        for known in _KNOWN_TOOLS:
            provider_obj = tool_registry.get_provider(known.name)
            has_cli = shutil.which(known.name) is not None
            has_acp = bool(provider_obj) and (
                has_cli or self._is_acp_provider_available(known.name, provider_obj)
            )
            if not has_cli and not has_acp:
                continue
            if known.name in seen:
                continue

            if provider_obj and has_acp:
                provider_type = "acp"
                supports_model = True
                model_optional = True
            else:
                provider_type = "cli"
                supports_model = False
                model_optional = False

            tools.append(
                AgentToolOption(
                    provider=provider_type,
                    tool_name=known.name,
                    display_name=known.display_name,
                    description=known.description,
                    supports_model=supports_model,
                    model_optional=model_optional,
                    skip_model_selection=False,
                ).__dict__
            )
            seen.add(known.name)

        if self.get_ttadk_tools():
            tools.append(
                AgentToolOption(
                    provider="ttadk",
                    tool_name="ttadk",
                    display_name="TTADK",
                    description="TTADK 多工具入口",
                    supports_model=False,
                ).__dict__
            )

        return self._sort_top_level_tools(tools)

    def _is_acp_provider_available(self, tool_name: str, provider_obj: object) -> bool:
        if not provider_obj:
            return False
        try:
            if tool_registry.get_availability(
                tool_name,
                allow_sync_probe=True,
                trigger_async_probe=False,
            ):
                return True
        except Exception:
            logger.debug("ACP availability check failed for %s", tool_name, exc_info=True)
        try:
            get_fallback = getattr(provider_obj, "get_fallback_command", None)
            return bool(callable(get_fallback) and get_fallback())
        except Exception:
            logger.debug("ACP fallback check failed for %s", tool_name, exc_info=True)
            return False

    def _sort_top_level_tools(self, tools: list[dict]) -> list[dict]:
        def key(item: dict) -> tuple[int, str]:
            priority = self._TOP_LEVEL_PRIORITY.get(
                (item.get("provider"), item.get("tool_name")),
                50,
            )
            name = str(item.get("display_name") or item.get("tool_name") or "")
            return priority, name

        return sorted(tools, key=key)

    def get_ttadk_tools(self) -> list[dict]:
        tools: list[dict] = []
        try:
            result = get_ttadk_manager().get_tools()
            for tool in result.tools:
                name = str(tool.name or "").strip()
                if not name:
                    continue
                tools.append(
                    AgentToolOption(
                        provider="ttadk",
                        tool_name=name,
                        display_name=name,
                        agent_name="ttadk",
                        description=f"TTADK · {name}",
                        supports_model=True,
                        model_optional=True,
                        skip_model_selection=getattr(
                            tool,
                            "skip_model_selection",
                            False,
                        ),
                    ).__dict__
                )
        except Exception:
            logger.debug("TTADK tool discovery failed", exc_info=True)
        return tools

    def get_models_for_tool(
        self,
        tool_name: str,
        provider: str = "ttadk",
        cwd: str | None = None,
        current_model: str | None = None,
        force_refresh: bool = True,
    ) -> list[dict]:
        """Return available models for an ACP or TTADK tool."""
        if provider == "acp":
            try:
                acp_models = fetch_acp_models(
                    tool_name,
                    cwd=cwd,
                    current_model=current_model,
                )
                models: list[dict] = []
                for model in acp_models:
                    item = {
                        "name": model.name,
                        "display_name": model.name,
                        "description": model.description or "",
                        "is_default": model.is_default,
                    }
                    reasoning_efforts = list(
                        getattr(model, "reasoning_efforts", ()) or ()
                    )
                    if reasoning_efforts:
                        item["reasoning_efforts"] = reasoning_efforts
                        item["adapted_reasoning_effort"] = getattr(
                            model,
                            "adapted_reasoning_effort",
                            None,
                        )
                    selection_variants = [
                        {
                            "name": variant.name,
                            "profile": variant.profile,
                            "effort": variant.effort,
                            "display_name": variant.display_name,
                            "is_variant_default": variant.is_variant_default,
                        }
                        for variant in (
                            getattr(model, "selection_variants", ()) or ()
                        )
                    ]
                    if selection_variants:
                        item["selection_variants"] = selection_variants
                    models.append(item)
                return models
            except Exception:
                return []

        try:
            models_result = get_ttadk_manager().get_models(
                tool_name=tool_name,
                cwd=cwd,
                force_refresh=force_refresh,
            )
            warnings = list(getattr(models_result, "warnings", []) or [])
            source = str(getattr(models_result, "source", "") or "").strip().lower()
            if source == "defaults" or "models_untrusted" in warnings:
                return []
            return [
                {
                    "name": model.name,
                    "display_name": getattr(model, "friendly_name", None)
                    or getattr(model, "display_name", None)
                    or model.name,
                    "is_default": getattr(model, "is_default", False),
                }
                for model in (models_result.models if models_result else [])
            ]
        except Exception:
            return []
