"""Shared dependency container for all message handlers.

HandlerContext bundles every collaborator that a handler might need so that
individual handlers don't require 16+ constructor parameters.  The FeishuWSClient
creates a single HandlerContext instance in its ``__init__`` and passes it to every
handler.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import lark_oapi as lark
    from lark_channel import FeishuChannel

    from ..access_control import IngressAccessPolicyProvider
    from ..acp.manager import ACPSessionManager
    from ..agent.intent_recognizer import IntentRecognizer
    from ..chat_lock import ChatLockManager
    from ..config.env_file_store import AtomicEnvFileStore
    from ..deep_engine import DeepEngineManager, ProgressReporter
    from ..mode import ModeManager
    from ..project import MessageProjectMapper, ProjectContextManager, ProjectManager
    from ..project.mapper import MessageLinker
    from ..repo_lock import RepoLockManager
    from ..spec_engine import SpecEngineManager, SpecReporter
    from ..tasking import TaskScheduler
    from ..thread import ThreadContextManager
    from ..trust.registry import ManagedGroupRegistry
    from ..workflow_engine.manager import WorkflowEngineManager
    from .image_handler import FeishuImageHandler


@dataclass
class HandlerContext:
    """All shared dependencies that handlers need, injected once."""

    settings: Any
    api_client_factory: Callable[[], "lark.Client"]
    message_callback: Callable[[str, str, str, Optional[str]], None]

    # Core services
    intent_recognizer: "IntentRecognizer"
    scheduler: "TaskScheduler"
    project_manager: "ProjectManager"
    message_mapper: "MessageProjectMapper"
    message_linker: "MessageLinker"
    mode_manager: "ModeManager"
    context_manager: "ProjectContextManager"
    deep_engine_manager: "DeepEngineManager"
    progress_reporter: "ProgressReporter"
    spec_engine_manager: "SpecEngineManager"
    spec_reporter: "SpecReporter"
    thread_manager: "ThreadContextManager"

    # Lazy-initialized singletons
    image_handler_factory: Callable[[], "FeishuImageHandler"]

    # Session managers (ACP/CLI-based; synced with `managers` in __post_init__)
    coco_manager: Optional["ACPSessionManager"] = None
    claude_manager: Optional["ACPSessionManager"] = None
    claude_w_manager: Optional["ACPSessionManager"] = None
    aiden_manager: Optional["ACPSessionManager"] = None
    codex_manager: Optional["ACPSessionManager"] = None
    codex_w_manager: Optional["ACPSessionManager"] = None
    gemini_manager: Optional["ACPSessionManager"] = None
    traex_manager: Optional["ACPSessionManager"] = None
    grok_manager: Optional["ACPSessionManager"] = None
    dsh_manager: Optional["ACPSessionManager"] = None
    agy_manager: Optional["ACPSessionManager"] = None
    agy_b_manager: Optional["ACPSessionManager"] = None

    # Shared mutable state
    working_dirs: dict[str, str] = field(default_factory=dict)
    working_dir_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_image_keys: dict[str, list[str]] = field(default_factory=dict)
    pending_image_lock: threading.Lock = field(default_factory=threading.Lock)
    enable_streaming: bool = True

    # Registry containers for decoupling
    managers: dict[str, "ACPSessionManager"] = field(default_factory=dict)
    handlers: dict[str, Any] = field(default_factory=dict)

    # Lock managers (multi-chat isolation)
    workflow_engine_manager: Optional["WorkflowEngineManager"] = None
    repo_lock_manager: Optional["RepoLockManager"] = None
    chat_lock_manager: Optional["ChatLockManager"] = None
    main_bot_outbound_audit: Optional[Callable[[str, str, str], None]] = None
    main_bot_outbound_audit_failure: Optional[Callable[[Exception], None]] = None
    tenant_key_resolver: Optional[Callable[[], str]] = None
    channel_client_factory: Optional[Callable[[], "FeishuChannel"]] = None
    ingress_access_policy_provider: Optional["IngressAccessPolicyProvider"] = None
    ingress_env_store: Optional["AtomicEnvFileStore"] = None
    managed_group_registry: Optional["ManagedGroupRegistry"] = None
    managed_group_owner_id: str = ""
    managed_group_receiving_bot_ref: str = ""
    managed_group_bot_rotation: Optional[Callable[[str], tuple[int, int]]] = None
    slash_permission_provider: Optional[Callable[[], Any]] = None

    def __post_init__(self) -> None:
        from ..mode.backends import ALL_BACKEND_METAS

        for spec in ALL_BACKEND_METAS:
            attr = spec.manager_attr
            field_val = getattr(self, attr, None)
            dict_val = self.managers.get(spec.mode_key)
            if field_val is None and dict_val is not None:
                setattr(self, attr, dict_val)
            elif field_val is not None and dict_val is None:
                self.managers[spec.mode_key] = field_val

    def get_manager(self, mode_key: str) -> Optional["ACPSessionManager"]:
        """Return the session manager for *mode_key*, checking attribute then registry."""
        mgr = getattr(self, f"{mode_key}_manager", None)
        if mgr is not None:
            return mgr
        return self.managers.get(mode_key)
