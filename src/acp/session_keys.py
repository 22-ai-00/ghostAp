"""Session key encoding and decoding helpers for ACP sessions."""

from __future__ import annotations


class SessionKeyCodec:
    """Encode the internal chat/project/thread routing key."""

    DEFAULT_PROJECT_PLACEHOLDER = "_default_"

    @classmethod
    def encode(
        cls,
        chat_id: str,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        project = project_id or cls.DEFAULT_PROJECT_PLACEHOLDER
        base = f"{chat_id}:{project}"
        return f"{base}:t:{thread_id}" if thread_id else base

    @classmethod
    def decode(cls, key: str) -> tuple[str, str | None, str | None]:
        try:
            value = str(key or "")
        except Exception:
            return "", None, None
        if not value:
            return "", None, None
        parts = value.split(":")
        chat_id = parts[0]
        project = parts[1] if len(parts) > 1 else ""
        project_id = (
            project if project and project != cls.DEFAULT_PROJECT_PLACEHOLDER else None
        )
        thread_id = parts[3] if len(parts) >= 4 and parts[2] == "t" else None
        return chat_id, project_id, thread_id
