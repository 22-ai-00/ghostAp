"""Request-scoped context for the Feishu message pipeline.

Every message handler receives a single RequestContext instead of 5+ scattered
parameters. This eliminates Optional[ProjectContext] threading and makes the
dispatch interface clean and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..mode import InteractionMode
    from ..project import ProjectContext
    from .slash_command_parser import CommandMatch


@dataclass
class RequestContext:
    """Full request-scoped context for a single incoming Feishu message.

    Created once at the ws_client ingress layer and passed through the
    entire dispatch pipeline. Replaces scattered (message_id, chat_id,
    text, project, command_match, ...) parameters.
    """

    # Core message fields
    message_id: str
    chat_id: str
    text: str
    chat_type: str = "group"

    # Resolved context (populated during ingress)
    project: Optional["ProjectContext"] = None
    command_match: Optional["CommandMatch"] = None

    # Mode state (populated by ws_client before dispatch)
    current_mode: Optional["InteractionMode"] = None
    is_in_programming: bool = False
    is_topic_engine_context: bool = False

    # Flags set during routing
    shell_fast_tracked: bool = False

    # Sender info
    sender_id: Optional[str] = None

    @property
    def project_id(self) -> Optional[str]:
        return self.project.project_id if self.project else None

    @property
    def has_project(self) -> bool:
        return self.project is not None

    @property
    def is_slash_command(self) -> bool:
        return (self.text or "").strip().startswith("/")


