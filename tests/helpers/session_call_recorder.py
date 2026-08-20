"""Test-only recorder for real session-manager and engine factory calls.

The recorder supplies lightweight sessions but observes the production manager
and engine factories.  It intentionally records topology, not elapsed time or
backend implementation details.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Callable

from src.acp.models import ACPEvent, PromptResult
from src.utils.retry import RetryPolicy, prompt_with_retry


@dataclass(frozen=True)
class FactoryCall:
    backend: str
    model: str | None
    cwd: str
    chat_id: str | None
    project_id: str | None
    thread_id: str | None
    session_key: str | None


@dataclass(frozen=True)
class PromptCall:
    backend: str
    model: str | None
    cwd: str
    chat_id: str | None
    project_id: str | None
    thread_id: str | None
    session_key: str | None
    prompt: str


@dataclass(frozen=True)
class RecordedEvent:
    sequence: int
    kind: str
    backend: str
    session_id: str


class RecordedSession:
    """Small SyncSession-compatible transport that records real caller input."""

    def __init__(self, recorder: "SessionCallRecorder", **factory_kwargs: Any) -> None:
        self._recorder = recorder
        self._factory_kwargs = factory_kwargs
        self.session_id = recorder.register_session(self)
        self.created_at = time.time()
        self.last_active = self.created_at
        self.message_count = 0
        self.last_query = ""
        self.cancel_count = 0
        self._force_dead = False
        self._fail_first_attempt = bool(factory_kwargs.get("fail_first_attempt", False))

    def describe_agent(self) -> str:
        return str(self._factory_kwargs.get("agent_type") or "recorded")

    def start(self, startup_timeout: float = 60) -> str:
        return self.session_id


    def load_local_history(self, session_id: str | None = None, limit: int = 200) -> list[dict]:
        return []

    def send_prompt(
        self,
        text: str,
        on_event: Callable[[ACPEvent], None] | None = None,
        timeout: int | None = None,
    ) -> PromptResult:
        self._recorder.record_event("prompt_attempt", self.describe_agent(), self.session_id)
        if self._fail_first_attempt:
            self._fail_first_attempt = False
            self._recorder.record_event("prompt_timeout", self.describe_agent(), self.session_id)
            raise TimeoutError("recorded transient timeout")
        self._recorder.record_prompt(self, text)
        self.last_query = text
        self.message_count += 1
        self.last_active = time.time()
        return PromptResult(stop_reason="end_turn", text="recorded completion")

    def send_prompt_with_retry(self, text: str, **kwargs: Any) -> PromptResult:
        backend = str(self._factory_kwargs.get("agent_type") or "")
        configured = kwargs.get("retry_policy") or RetryPolicy()
        policy = RetryPolicy(
            max_retries=configured.max_retries,
            retry_delay=0,
            backoff_multiplier=configured.backoff_multiplier,
            max_delay=0,
            jitter_factor=0,
            total_timeout=configured.total_timeout,
        )
        before = len(self._recorder.events)
        result = prompt_with_retry(
            lambda: self.send_prompt(
                text,
                on_event=kwargs.get("on_event"),
                timeout=kwargs.get("timeout"),
            ),
            Event(),
            retry_policy=policy,
            before_retry=kwargs.get("before_retry"),
            total_timeout=kwargs.get("total_timeout"),
        )
        attempts = sum(
            event.kind == "prompt_attempt"
            for event in self._recorder.events[before:]
        )
        self._recorder.record_retry(backend, attempts, self.session_id)
        return result

    def set_model(self, model_name: str) -> bool:
        """Mirror the ACP protocol switch without recreating the session."""
        self._factory_kwargs["model_name"] = model_name
        self._recorder.update_session_model(self, model_name)
        self._recorder.record_event("set_model", self.describe_agent(), self.session_id)
        return True

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool:
        self.cancel_count += 1
        self._recorder.record_cancel(self)
        return True

    def close(self) -> None:
        return None

    def to_snapshot(self) -> dict:
        return {"session_id": self.session_id}

    def get_session_info(self) -> str:
        return self.session_id

    def is_server_running(self) -> bool:
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return True


class SessionCallRecorder:
    """Collect the remote-call topology observed through production seams."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.factory_calls: list[FactoryCall] = []
        self.prompt_calls: list[PromptCall] = []
        self.sessions: list[RecordedSession] = []
        self.cancelled_sessions: list[RecordedSession] = []
        self.retry_calls: list[tuple[str, int]] = []
        self.events: list[RecordedEvent] = []
        self._session_context: dict[int, dict[str, Any]] = {}
        self._factory_call_index: dict[int, int] = {}
        self.fail_first_prompt = False

    def record_event(self, kind: str, backend: str, session_id: str) -> None:
        with self._lock:
            self.events.append(RecordedEvent(len(self.events) + 1, kind, backend, session_id))

    def register_session(self, session: RecordedSession) -> str:
        with self._lock:
            self.sessions.append(session)
            return f"recorded-{len(self.sessions)}"

    def session_factory(self, *, agent_type: str, cwd: str, model_name: str | None = None, **kwargs: Any):
        session = self._create_session(agent_type, cwd=cwd, model_name=model_name, **kwargs)
        return session, session.session_id, {"source": "session-call-recorder"}

    def factory_for_backend(self, backend: str) -> Callable[..., RecordedSession]:
        """Return a callable usable in place of a production session class."""

        def factory(*args: Any, **kwargs: Any) -> RecordedSession:
            kwargs.pop("agent_type", None)
            return self._create_session(
                backend,
                cwd=str(kwargs.pop("cwd", ".") or "."),
                model_name=kwargs.pop("model_name", None),
                **kwargs,
            )

        return factory

    def _create_session(
        self,
        backend: str,
        *,
        cwd: str,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> RecordedSession:
        fail_first_attempt = kwargs.pop("fail_first_attempt", self.fail_first_prompt)
        session = RecordedSession(
            self,
            agent_type=backend,
            cwd=cwd,
            model_name=model_name,
            **kwargs,
            fail_first_attempt=fail_first_attempt,
        )
        with self._lock:
            self._session_context[id(session)] = {
                "backend": backend,
                "model": model_name,
                "cwd": cwd,
                "chat_id": None,
                "project_id": kwargs.get("project_id"),
                "thread_id": None,
                "session_key": None,
            }
            self._factory_call_index[id(session)] = len(self.factory_calls)
            self.factory_calls.append(FactoryCall(**self._session_context[id(session)]))
        self.record_event("factory", backend, session.session_id)
        return session

    def observe_manager_session_key(
        self,
        session: RecordedSession,
        *,
        chat_id: str,
        project_id: str | None,
        thread_id: str | None,
        session_key: str,
    ) -> None:
        """Enrich a factory record from the manager's already-created key.

        Factory kwargs do not carry the opaque key.  This reads the actual
        manager key after startup; it does not inject or choose a key.
        """
        with self._lock:
            context = self._session_context[id(session)]
            context.update(
                chat_id=chat_id,
                project_id=project_id,
                thread_id=thread_id,
                session_key=session_key,
            )
            self.factory_calls[self._factory_call_index[id(session)]] = FactoryCall(**context)

    def record_prompt(self, session: RecordedSession, prompt: str) -> None:
        with self._lock:
            self.prompt_calls.append(PromptCall(**self._session_context[id(session)], prompt=prompt))
        self.record_event("prompt", session.describe_agent(), session.session_id)

    def update_session_model(self, session: RecordedSession, model_name: str) -> None:
        with self._lock:
            self._session_context[id(session)]["model"] = model_name
            index = self._factory_call_index[id(session)]
            self.factory_calls[index] = FactoryCall(**self._session_context[id(session)])

    def record_retry(self, backend: str, attempts: int, session_id: str) -> None:
        with self._lock:
            self.retry_calls.append((backend, attempts))
        self.record_event("retry_complete", backend, session_id)

    def record_cancel(self, session: RecordedSession) -> None:
        with self._lock:
            self.cancelled_sessions.append(session)
        self.record_event("cancel", session.describe_agent(), session.session_id)

    def remote_call_topology(self) -> tuple[str, ...]:
        return tuple(
            f"{event.kind}:{event.backend}"
            for event in self.events
            if event.kind in {"factory", "prompt"}
        )
