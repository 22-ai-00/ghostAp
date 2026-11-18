"""Command routing primitives for the Feishu message pipeline."""

from __future__ import annotations


class CommandRouter:
    """Stateless command detection — replaces ws_client._is_*_command methods."""

    @staticmethod
    def is_exit_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_exit_command(text)

    @staticmethod
    def is_deep_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_deep_command(text)

    @staticmethod
    def is_spec_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_spec_command(text)

    @staticmethod
    def is_workflow_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_workflow_command(text)
