"""Feishu Slash Command catalog for the main GhostAP Bot.

Slash Command registration only controls the command-discovery panel. Selected
commands still arrive through Channel SDK as ordinary message events and are
executed by the existing request-scoped SlashCommandParser routing chain.
"""

from __future__ import annotations

from typing import Protocol

from lark_oapi.core.model.base_request import BaseRequest
from lark_oapi.core.model.base_response import BaseResponse

from ..autonomous.provisioning.slash_commands import (
    SlashCommand,
    SlashCommandReconciler,
    VerifiedSlashState,
)
from ..autonomous.provisioning.slash_lark import LarkSlashCommandAPI
from .product_catalog import get_slash_discoverable_actions


class _AsyncLarkClient(Protocol):
    async def arequest(self, request: BaseRequest) -> BaseResponse: ...


# Register primary public spellings rather than every compatibility alias.  The
# typed product catalog is the only command list; this module is its Feishu
# discovery-panel projection.
MAIN_AGENT_COMMANDS: tuple[SlashCommand, ...] = tuple(
    SlashCommand(action.command, action.description, action.usage)
    for action in get_slash_discoverable_actions()
)


async def reconcile_main_agent_slash_commands(
    client: _AsyncLarkClient,
    *,
    app_id: str,
) -> VerifiedSlashState:
    """Converge the main Bot's server-side Slash Command panel."""

    return await SlashCommandReconciler(
        LarkSlashCommandAPI(client, expected_app_id=app_id),
        desired=MAIN_AGENT_COMMANDS,
    ).reconcile()
