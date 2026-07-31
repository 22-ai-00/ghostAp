"""Error types for project_chat module."""

from enum import Enum


class ProjectChatError(Exception):
    """Base error for project chat operations."""
    pass


class CreateChatFailureDisposition(str, Enum):
    DEFINITIVE_REJECTED = "definitive_rejected"
    OUTCOME_UNKNOWN = "outcome_unknown"


class CreateChatError(ProjectChatError):
    """Failed create call with an explicit remote-outcome disposition."""

    def __init__(
        self,
        message: str,
        *,
        disposition: CreateChatFailureDisposition = (
            CreateChatFailureDisposition.OUTCOME_UNKNOWN
        ),
        api_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.disposition = disposition
        self.api_code = api_code


class BindError(ProjectChatError):
    """Failed to bind project to chat."""
    pass
