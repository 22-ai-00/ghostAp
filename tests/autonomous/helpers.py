from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from src.autonomous.context import ContextMessage, MessagePage, ResolvedThread
from src.autonomous.runtime.employee_session import (
    EmployeeSessionBootstrap,
    EmployeeSessionKey,
)


def make_context_message(
    message_id: str,
    text: str,
    *,
    create: int,
    position: int,
) -> ContextMessage:
    return ContextMessage(
        message_id=message_id,
        sender_id="ou_user",
        sender_type="user",
        text=text,
        timestamp=create / 1000,
        is_current=message_id == "om_current",
        chat_id="oc_1",
        thread_id="omt_1",
        root_id="om_root",
        sender_id_type="open_id",
        sender_tenant_key="tenant_1",
        create_time_ms=create,
        update_time_ms=create,
        message_position=position,
        thread_message_position=position,
    )


def message_pages(messages) -> list[MessagePage]:
    return [MessagePage(tuple(messages), False)]


def stable_thread() -> list[ContextMessage]:
    return [
        make_context_message("om_root", "root", create=1_000, position=0),
        make_context_message("om_before", "before", create=2_000, position=1),
        make_context_message("om_current", "current", create=3_000, position=2),
    ]


class FakeEmployeeMessageSource:
    def __init__(self, *, traversals: list[list[MessagePage]]) -> None:
        self.scope = None
        self._traversals = traversals
        self._traversal = -1
        self.thread_calls = 0
        self.chat_calls = 0
        self.reset_calls = 0
        self.closed = False

    def resolve_thread(self) -> ResolvedThread:
        assert self.scope is not None
        return ResolvedThread(
            self.scope.thread_root_message_id,
            self.scope.feishu_thread_id,
            self.scope.current_message_id,
        )

    def list_thread_messages(
        self,
        *,
        page_token: str = "",
        page_size: int = 50,
    ) -> MessagePage:
        del page_size
        self.thread_calls += 1
        if not page_token:
            self._traversal += 1
            page_index = 0
        else:
            page_index = int(page_token)
        pages = self._traversals[self._traversal]
        return pages[page_index]

    def list_chat_messages(
        self,
        *,
        page_token: str = "",
        page_size: int = 20,
    ) -> MessagePage:
        del page_token, page_size
        self.chat_calls += 1
        return MessagePage((), False)

    def reset_chat_traversal(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def make_employee_bootstrap(tmp_path: Path) -> EmployeeSessionBootstrap:
    project_root = str(tmp_path / "project")
    workspace_root = str(tmp_path / "employee" / "workspace")
    codex_home = str(tmp_path / "employee" / "runtime" / "codex-home")
    instruction_text = "Act as the assigned employee."
    instruction_digest = hashlib.sha256(instruction_text.encode()).hexdigest()
    return EmployeeSessionBootstrap(
        session_key=EmployeeSessionKey(
            tenant_key="tenant_1",
            agent_id="agt_1",
            project_root=project_root,
            backend="codex",
            model="m",
            profile="",
            effort="",
            identity_version=1,
            instruction_digest=instruction_digest,
        ),
        project_root=project_root,
        workspace_root=workspace_root,
        codex_home=codex_home,
        instruction_text=instruction_text,
        instruction_digest=instruction_digest,
        read_only_roots=(workspace_root,),
        writable_roots=(project_root,),
    )


class EmployeeTestSession:
    def is_server_healthy(self) -> bool:
        return True

    def send_prompt(self, _prompt: str, *, timeout: float):
        del timeout
        return SimpleNamespace(text="employee result")

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None
