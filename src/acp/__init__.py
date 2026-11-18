"""ACP (Agent Client Protocol) integration layer for GhostAP.

Provides structured communication with AI agents (Coco/Claude) via JSON-RPC 2.0
over stdio, replacing the previous subprocess-based CLI interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import GhostAPClient
    from .continuation import PromptContinuationResult, run_prompt_with_continuation
    from .execution_windows import run_prompt_across_execution_windows
    from .finalization import run_prompt_with_finalization
    from .manager import ACPSessionManager, AgentSessionManager
    from .models import (
        ACPEvent,
        ACPEventType,
        ACPImageInfo,
        ACPSessionState,
        PlanEntryInfo,
        PlanInfo,
        PromptResult,
        ToolCallInfo,
    )
    from .renderer import ACPEventRenderer
    from .session import ACPResumeRejected, ACPSession
    from .sync_adapter import SyncACPSession, start_session_with_retry

_MODEL_EXPORTS = frozenset(
    {
        "ACPEvent",
        "ACPEventType",
        "ACPImageInfo",
        "ACPSessionState",
        "PlanEntryInfo",
        "PlanInfo",
        "PromptResult",
        "ToolCallInfo",
    }
)


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import models as _models

        for attr in _MODEL_EXPORTS:
            globals()[attr] = getattr(_models, attr)
        return globals()[name]
    if name in ("PromptContinuationResult", "run_prompt_with_continuation"):
        from .continuation import PromptContinuationResult, run_prompt_with_continuation

        globals()["PromptContinuationResult"] = PromptContinuationResult
        globals()["run_prompt_with_continuation"] = run_prompt_with_continuation
        return globals()[name]
    if name == "run_prompt_across_execution_windows":
        from .execution_windows import run_prompt_across_execution_windows

        globals()["run_prompt_across_execution_windows"] = run_prompt_across_execution_windows
        return run_prompt_across_execution_windows
    if name == "run_prompt_with_finalization":
        from .finalization import run_prompt_with_finalization

        globals()["run_prompt_with_finalization"] = run_prompt_with_finalization
        return run_prompt_with_finalization
    if name == "ACPEventRenderer":
        from .renderer import ACPEventRenderer

        globals()["ACPEventRenderer"] = ACPEventRenderer
        return ACPEventRenderer
    if name == "GhostAPClient":
        from .client import GhostAPClient

        globals()["GhostAPClient"] = GhostAPClient
        return GhostAPClient
    if name in ("ACPResumeRejected", "ACPSession"):
        from .session import ACPResumeRejected, ACPSession

        globals()["ACPResumeRejected"] = ACPResumeRejected
        globals()["ACPSession"] = ACPSession
        return globals()[name]
    if name in ("SyncACPSession", "start_session_with_retry"):
        from .sync_adapter import SyncACPSession, start_session_with_retry

        globals()["SyncACPSession"] = SyncACPSession
        globals()["start_session_with_retry"] = start_session_with_retry
        return globals()[name]
    if name in ("ACPSessionManager", "AgentSessionManager"):
        from .manager import ACPSessionManager, AgentSessionManager

        globals()["ACPSessionManager"] = ACPSessionManager
        globals()["AgentSessionManager"] = AgentSessionManager
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACPEvent",
    "ACPEventType",
    "ACPImageInfo",
    "ACPSessionState",
    "PlanEntryInfo",
    "PlanInfo",
    "PromptResult",
    "ToolCallInfo",
    "GhostAPClient",
    "ACPSession",
    "ACPResumeRejected",
    "SyncACPSession",
    "start_session_with_retry",
    "PromptContinuationResult",
    "run_prompt_with_continuation",
    "run_prompt_with_finalization",
    "run_prompt_across_execution_windows",
    "ACPEventRenderer",
    "ACPSessionManager",
    "AgentSessionManager",
]
