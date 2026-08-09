"""Production composition root for the autonomous work system.

Uses lark-oapi for REST API calls and lark-channel-sdk for WebSocket
event subscription. Wires all layers together at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lark_oapi as lark

logger = logging.getLogger(__name__)


class AutonomousContainer:
    """Legacy standalone composition shell; not wired into production routing.

    Production autonomous work is composed by ``EmployeeDepartmentRuntime`` and
    entered from Team. This compatibility shell remains importable for older
    callers, but its Manager command surface is deliberately retired.

    Uses:
    - lark-oapi (REST): message delivery, card updates, bot management
    - lark-channel-sdk (WebSocket): event subscription for durable ingress
    """

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        mode: str = "off",
        app_id: str = "",
        app_secret: str = "",
    ) -> None:
        self._data_dir = data_dir or Path("data/autonomous")
        self._mode = mode
        self._app_id = app_id
        self._app_secret = app_secret
        self._started = False
        self._lark_client: lark.Client | None = None
        self._supervisor: Any = None
        self._manager_handler: Any = None
        self._feishu_adapter: Any = None
        self._coordinator: Any = None
        self._employee_manager: Any = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def supervisor(self) -> Any:
        return self._supervisor

    @property
    def manager_handler(self) -> Any:
        return self._manager_handler

    @property
    def feishu_adapter(self) -> Any:
        return self._feishu_adapter

    @property
    def coordinator(self) -> Any:
        return self._coordinator

    @property
    def employee_manager(self) -> Any:
        return self._employee_manager

    async def start(self) -> None:
        if self._mode == "off":
            logger.info("autonomous system disabled (mode=off)")
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)

        if self._app_id and self._app_secret:
            self._lark_client = lark.Client.builder() \
                .app_id(self._app_id) \
                .app_secret(self._app_secret) \
                .build()
            logger.info("lark-oapi client initialized for autonomous system")

            from .manager.feishu_adapter import FeishuAdapter
            self._feishu_adapter = FeishuAdapter(
                app_id=self._app_id,
                app_secret=self._app_secret,
                lark_client=self._lark_client,
            )

        logger.info("autonomous system starting (mode=%s)", self._mode)
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        logger.info("autonomous system shutting down")
        self._started = False
        self._lark_client = None

    def health_check(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "started": self._started,
            "data_dir": str(self._data_dir),
            "lark_client_ready": self._lark_client is not None,
        }

    async def handle_employee_creation(
        self,
        *,
        chat_id: str,
        role: str,
        tool: str,
        model: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Handle employee creation request from Feishu card callback.

        Called when user completes the employee creation card interaction.
        """
        if not self._employee_manager:
            from .employees import EmployeeManager

            class _NullJournal:
                async def write_event(self, event_type: str, payload: dict) -> None:
                    pass

            self._employee_manager = EmployeeManager(journal=_NullJournal())

        employee = await self._employee_manager.hire(
            name=f"{role}_{tool}",
            role=role,
            tool=tool,
            model=model,
        )

        if self._feishu_adapter:
            from .manager.cards import build_employee_created_card
            card = build_employee_created_card(
                employee_id=employee.employee_id,
                name=employee.name,
                role=employee.role,
                tool=employee.tool,
                model=employee.model,
                worker_type=employee.worker_type.value,
            )
            await self._feishu_adapter.send_card(chat_id, card)

        return {
            "employee_id": employee.employee_id,
            "name": employee.name,
            "role": role,
            "tool": tool,
            "model": model,
        }
