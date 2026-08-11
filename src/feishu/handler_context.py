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
    from ..autonomous.data.composition import EmployeeDataComposition
    from ..autonomous.membership.service import EmployeeMembershipService
    from ..autonomous.provisioning.fire_service import EmployeeFireService
    from ..autonomous.provisioning.hire_port import EmployeeHireService
    from ..autonomous.team import EmployeeTeamService
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

    # Session managers (ACP-based)
    coco_manager: "ACPSessionManager"
    claude_manager: "ACPSessionManager"
    aiden_manager: "ACPSessionManager"
    codex_manager: "ACPSessionManager"
    gemini_manager: "ACPSessionManager"
    traex_manager: "ACPSessionManager"
    grok_manager: "ACPSessionManager"

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
    employee_hire_service: Optional["EmployeeHireService"] = None
    employee_fire_service: Optional["EmployeeFireService"] = None
    employee_hire_readiness: Optional[Callable[[], Any]] = None
    employee_membership_service: Optional["EmployeeMembershipService"] = None
    employee_data_composition: Optional["EmployeeDataComposition"] = None
    employee_team_service: Optional["EmployeeTeamService"] = None
    employee_runtime_facade: Any = None
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
