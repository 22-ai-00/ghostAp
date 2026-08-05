"""Session backends abstraction package.

GhostAP currently supports two different ways to talk to an agent:

1) ACP backend (JSON-RPC 2.0 over stdio) — used by Coco.
2) CLI backend (spawn per prompt)       — used by Claude Code CLI.

The handlers expect an ACP-like streaming callback signature. For CLI backend
we downgrade to text-only ACPEvent(TEXT_CHUNK) events so that existing
rendering and streaming cards can be reused.
"""

from __future__ import annotations

# stdlib re-exports: required for backward-compat test patching
# (e.g. ``patch("src.agent_session.subprocess.Popen")``)
import subprocess  # noqa: F401
import uuid  # noqa: F401

from .backend_resolver import is_cli_backend, is_ttadk_type, resolve_backend_kind, resolve_cwd
from .claude_cli import ClaudeCLIConfig, SyncClaudeCLISession
from .factory import (
    EphemeralReviewSession,
    close_session_safely,
    create_auxiliary_session,
    create_engine_session,
    create_review_session,
    create_sync_session,
    resolve_ttadk_engine_startup_model,
)
from .model_diagnostics import (
    _apply_compaction_once,
    _build_generic_error_blob,
    _detect_rate_limit,
    _extract_model_from_agent_args,
    _remove_model_in_agent_args,
    _replace_model_in_agent_args,
    classify_model_failure,
)
from .protocol import SyncSession
from .tool_permissions import (
    AuxiliarySessionPermissionError,
    AuxiliaryToolProfile,
    apply_auxiliary_tool_profile,
    deny_all_tools,
)
from .ttadk_cli import (
    SyncTTADKCLISession,
    _build_ttadk_passthrough_prompt,
    _is_ttadk_preamble_line,
    _JSONTextExtractor,
)
from .wrappers import ModelFailureAwareSession, RateLimitAwareSession

__all__ = [
    "SyncSession", "ClaudeCLIConfig", "SyncClaudeCLISession", "SyncTTADKCLISession",
    "RateLimitAwareSession", "ModelFailureAwareSession", "EphemeralReviewSession",
    "classify_model_failure", "create_sync_session", "create_engine_session",
    "create_auxiliary_session",
    "create_review_session",
    "close_session_safely", "resolve_ttadk_engine_startup_model",
    "_JSONTextExtractor", "_is_ttadk_preamble_line", "_build_ttadk_passthrough_prompt",
    "_detect_rate_limit", "_extract_model_from_agent_args", "_replace_model_in_agent_args",
    "_remove_model_in_agent_args", "_apply_compaction_once", "_build_generic_error_blob",
    "resolve_backend_kind", "is_cli_backend", "is_ttadk_type", "resolve_cwd",
    "AuxiliarySessionPermissionError", "AuxiliaryToolProfile",
    "apply_auxiliary_tool_profile", "deny_all_tools",
]
