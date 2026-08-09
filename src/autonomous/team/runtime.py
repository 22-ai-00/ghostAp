"""Standalone Team binding for employee execution."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterator

from ..runtime.session_host import EmployeeSessionHost


def _stable_identity(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


class TeamRuntimeResolutionError(RuntimeError):
    """A Team execution coordinate cannot be resolved."""


@contextmanager
def team_runtime_guard() -> Iterator[None]:
    yield


@dataclass(frozen=True, slots=True)
class TeamExecutionBinding:
    engine_identity: str
    chat_id: str
    root_identity: str
    canonical_root: str
    engine: EmployeeSessionHost


class TeamRuntime:
    """Resolve direct Team execution without chat activation or mode state."""

    def __init__(
        self,
        *,
        project_root_resolver: Callable[[str], str] | None = None,
        owner_resolver: Callable[[str], str] | None = None,
        session_host: EmployeeSessionHost | None = None,
    ) -> None:
        self._project_root_resolver = project_root_resolver or (lambda _chat_id: os.getcwd())
        self._owner_resolver = owner_resolver or (lambda _chat_id: "")
        self._session_host = session_host or EmployeeSessionHost()

    def resolve_employee_engine(self, *, chat_id: str) -> TeamExecutionBinding:
        if not isinstance(chat_id, str) or not chat_id:
            raise TeamRuntimeResolutionError("team chat coordinate is required")
        resolved_root = self._project_root_resolver(chat_id)
        if not isinstance(resolved_root, str) or not resolved_root:
            raise TeamRuntimeResolutionError("team project root is unavailable")
        root = os.path.realpath(resolved_root)
        return TeamExecutionBinding(
            engine_identity=_stable_identity("team", chat_id),
            chat_id=chat_id,
            root_identity=_stable_identity("root", root),
            canonical_root=root,
            engine=self._session_host,
        )

    @contextmanager
    def employee_activation_guard(self, *, chat_id: str) -> Iterator[TeamExecutionBinding]:
        yield self.resolve_employee_engine(chat_id=chat_id)

    def get_activated_engine(self, chat_id: str):
        binding = self.resolve_employee_engine(chat_id=chat_id)
        return SimpleNamespace(
            channel=SimpleNamespace(owner_id=self._owner_resolver(chat_id)),
            root_path=binding.canonical_root,
        )

    def close(self) -> None:
        self._session_host.close()


__all__ = [
    "TeamExecutionBinding",
    "TeamRuntime",
    "TeamRuntimeResolutionError",
    "team_runtime_guard",
]
