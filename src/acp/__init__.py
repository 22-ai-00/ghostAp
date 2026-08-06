"""ACP (Agent Client Protocol) integration layer for GhostAP.

Provides structured communication with AI agents (Coco/Claude) via JSON-RPC 2.0
over stdio, replacing the previous subprocess-based CLI interaction.
"""

from .client import GhostAPClient
from .continuation import (
    PromptContinuationResult,
    run_prompt_with_continuation,
)
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
from .session import ACPResumeRejected, ACPSession
from .sync_adapter import SyncACPSession, start_session_with_retry

# NOTE: .manager import is deferred via __getattr__ (PEP 562) to break a
# circular dependency with the agent_session package.  agent_session submodules
# import from acp.models / acp.sync_adapter at module level, and acp.manager
# imports from agent_session — eagerly loading .manager here would trigger the
# cycle when agent_session is initialised first.


def __getattr__(name: str):
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
    "ACPEventRenderer",
    "ACPSessionManager",
    "AgentSessionManager",
]
