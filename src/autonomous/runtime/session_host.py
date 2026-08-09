"""Standalone reusable ACP sessions for persistent employees."""

from __future__ import annotations

import os
import threading

from src.agent_session.factory import create_engine_session, employee_session_environment

from ..workforce.identity import AgentIdentity


class EmployeeSessionUnavailableError(RuntimeError):
    """An employee backend session cannot be opened safely."""


class _EmployeeSessionLease:
    def __init__(self, host: "EmployeeSessionHost", agent_id: str, session: object) -> None:
        self._host = host
        self._agent_id = agent_id
        self._session = session
        self._closed = False

    def send_prompt(self, prompt: str, *, timeout: float):
        return self._session.send_prompt(prompt, timeout=timeout)

    def is_server_healthy(self) -> bool:
        probe = getattr(self._session, "is_server_healthy", None)
        return bool(probe()) if callable(probe) else not self._closed

    def cancel(self) -> None:
        cancel = getattr(self._session, "cancel", None)
        if callable(cancel):
            cancel()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._host._release(self._agent_id, self._session)
        close = getattr(self._session, "close", None)
        if callable(close):
            close()


class EmployeeSessionHost:
    """Open employee-scoped ACP sessions without a chat execution engine."""

    def __init__(self) -> None:
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._sessions: dict[str, object] = {}

    @staticmethod
    def _workspace(agent: AgentIdentity) -> str:
        raw = agent.workspace_path
        if (
            not isinstance(raw, str)
            or not raw
            or "\x00" in raw
            or not os.path.isabs(raw)
        ):
            raise EmployeeSessionUnavailableError("employee workspace is unavailable")
        workspace = os.path.realpath(raw)
        if not os.path.isdir(workspace):
            raise EmployeeSessionUnavailableError("employee workspace is unavailable")
        return workspace

    @staticmethod
    def _under(path: str, roots: tuple[str, ...]) -> bool:
        candidate = os.path.realpath(path)
        return any(candidate == root or candidate.startswith(root + os.sep) for root in roots)

    def _install_policy(self, session: object, agent: AgentIdentity, workspace: str) -> None:
        set_filter = getattr(session, "set_tool_filter", None)
        if not callable(set_filter):
            raise EmployeeSessionUnavailableError("employee backend lacks tool filtering")
        permissions = set(agent.permissions)
        capabilities = set(agent.capabilities)
        roots = (workspace,)
        configure = getattr(session, "configure_employee_sandbox", None)
        writable = roots if "file_write" in permissions and "file_write" in capabilities else ()
        if callable(configure):
            configure(read_only_roots=roots, writable_roots=writable)

        def tool_filter(tool_name: str, args: dict | None = None) -> bool:
            args = args or {}
            name = (tool_name or "").lower()
            if name == "shell":
                return "shell" in permissions and "shell" in capabilities
            if name == "git":
                return "git" in permissions and "git" in capabilities
            if name in {"file_read", "file_list", "grep", "search"}:
                path = str(args.get("path") or args.get("file_path") or workspace)
                path = path if os.path.isabs(path) else os.path.join(workspace, path)
                return bool({"file_read", "shell", "git"} & permissions) and self._under(path, roots)
            if name == "file_write":
                path = str(args.get("path") or args.get("file_path") or "")
                path = path if os.path.isabs(path) else os.path.join(workspace, path)
                return (
                    bool(path)
                    and "file_write" in permissions
                    and "file_write" in capabilities
                    and self._under(path, writable)
                )
            return False

        set_filter(tool_filter)

    def open_employee_session(self, agent: AgentIdentity, *, env: dict[str, str]) -> _EmployeeSessionLease:
        if agent.security_profile != "employee_v1" or not isinstance(env, dict):
            raise EmployeeSessionUnavailableError("employee session authority is invalid")
        workspace = self._workspace(agent)
        with employee_session_environment(env):
            session = create_engine_session(
                agent_type=agent.agent_type,
                cwd=workspace,
                model_name=agent.model_name or None,
                thread_id=f"employee_actor_{agent.agent_id}",
                auto_approve=True,
                require_tool_filter=True,
            )
        if session is None:
            raise EmployeeSessionUnavailableError("employee backend session creation failed")
        try:
            self._install_policy(session, agent, workspace)
        except Exception:
            close = getattr(session, "close", None)
            if callable(close):
                close()
            raise
        with self._lock:
            previous = self._sessions.get(agent.agent_id)
            if previous is not None and previous is not session:
                close = getattr(session, "close", None)
                if callable(close):
                    close()
                raise EmployeeSessionUnavailableError("employee already has an active session")
            self._sessions[agent.agent_id] = session
        return _EmployeeSessionLease(self, agent.agent_id, session)

    def run_agent_session(
        self,
        agent: AgentIdentity,
        prompt: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        if env is None:
            raise EmployeeSessionUnavailableError("employee environment is required")
        lease = self.open_employee_session(agent, env=env)
        try:
            result = lease.send_prompt(prompt, timeout=float(timeout or 600.0))
            if not isinstance(result, str) or not result:
                raise EmployeeSessionUnavailableError("employee backend returned no output")
            return result
        finally:
            lease.close()

    def preview_employee_session_prompt(self, agent: AgentIdentity, prompt: str) -> str:
        from .employee_session import EmployeeSessionBootstrap
        return EmployeeSessionBootstrap.from_agent(
            tenant_key="employee",
            agent=agent,
            project_root=agent.workspace_path,
            identity_version=0,
        ).wrap_prompt(prompt)

    def cancel_employee_session(self, agent_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(agent_id, None)
        if session is None:
            return False
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            cancel()
        return True

    stop_agent = cancel_employee_session

    def _release(self, agent_id: str, session: object) -> None:
        with self._lock:
            if self._sessions.get(agent_id) is session:
                del self._sessions[agent_id]

    def close(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            close = getattr(session, "close", None)
            if callable(close):
                close()


__all__ = ["EmployeeSessionHost", "EmployeeSessionUnavailableError"]
