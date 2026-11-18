"""ACP (Agent Client Protocol) integration layer for GhostAP.

Provides structured communication with AI agents (Coco/Claude) via JSON-RPC 2.0
over stdio, replacing the previous subprocess-based CLI interaction.
"""

from .continuation import (
    PromptContinuationResult,
    run_prompt_with_continuation,
)
from .execution_windows import run_prompt_across_execution_windows
from .finalization import run_prompt_with_finalization
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

# NOTE: .client, .session, .sync_adapter, and .manager imports are deferred via
# __getattr__ (PEP 562).  .client/.session/.sync_adapter pull in the external
# `acp` (agent-client-protocol) SDK (~9 MB USS), which is only needed when an
# ACP session or probe actually starts; deferring .manager also breaks the
# circular dependency with the agent_session package.


def __getattr__(name: str):
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
