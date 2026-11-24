"""Shared transport-boundary E2E harness for GhostAP Workflow (/wf) tests.

Only external process/network boundaries (``create_engine_session``,
``subprocess.Popen``, ACP transport, Feishu IM HTTP/COT APIs, and the task
scheduler) are replaced here. All routing, Card 2.0 normalization/guards,
WorkflowEngine, RuntimeBridge + Node.js runtime.js, AgentExecutor,
WorkflowJournal, WorkflowStateManager, and ProjectManager persistence stay real.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from lark_oapi.core.model.base_response import BaseResponse
from lark_oapi.core.model.raw_response import RawResponse

from src.acp.models import ACPEvent, PromptResult
from src.card.schema import require_card_json_2
from src.card.shared.truncation import (
    _count_explicit_table_nodes,
    _count_markdown_tables_in_payload,
    count_tagged_nodes,
)
from src.config.settings import Settings
from src.feishu.ws_client import FeishuWSClient
from src.tasking.scheduler import TaskStatus
from src.workflow_engine.engine import WorkflowEngine, WorkflowEngineCallbacks
from src.workflow_engine.models import WorkflowProject
from src.workflow_engine.progress_coalescer import ProgressCoalescer
from src.workflow_engine.reporting import WorkflowReportFiles, write_workflow_report_files
from src.workflow_engine.run_spec import WorkflowRunSpec


@dataclass(frozen=True)
class SessionInvocation:
    """One recorded ``create_engine_session`` + ``send_prompt`` call."""

    sequence: int
    agent_type: str
    model_name: str | None
    cwd: str
    kwargs: dict[str, Any]
    prompt: str
    turn_index: int
    thread_name: str


@dataclass
class ScriptedRule:
    """Deterministic behavior rule for a ``ScriptedAgentSession``."""

    output: str | Callable[[str, int, SessionInvocation], str] = "ok"
    events: tuple[ACPEvent, ...] = ()
    raise_on_start: BaseException | None = None
    raise_on_prompt: BaseException | None = None
    output_tokens: int = 12
    stop_reason: str = "end_turn"
    block_event: threading.Event | None = None
    entered_event: threading.Event | None = None
    barrier: threading.Barrier | None = None
    cancel_wait_s: float = 0.0


class ScriptedAgentSession:
    """SyncSession-compatible fake at the LLM/process boundary."""

    def __init__(
        self,
        recorder: "WorkflowSessionRecorder",
        *,
        agent_type: str,
        cwd: str,
        model_name: str | None,
        cancel_event: threading.Event | None,
        kwargs: dict[str, Any],
    ) -> None:
        self._recorder = recorder
        self._agent_type = agent_type
        self._cwd = cwd
        self._model_name = model_name
        self._cancel_event = cancel_event
        self._kwargs = dict(kwargs)
        self._turn_index = 0
        self.closed = False
        self.cancelled = False
        self.session_id = f"scripted-{agent_type}-{recorder.next_session_num()}"

    def start(self, startup_timeout: float = 20.0) -> str:
        del startup_timeout
        return self.session_id

    def is_server_running(self) -> bool:
        return not self.closed

    def send_prompt(
        self,
        text: str,
        on_event: Callable[[ACPEvent], None] | None = None,
        timeout: int | None = None,
        **_kwargs: Any,
    ) -> PromptResult:
        del timeout
        self._turn_index += 1
        inv = self._recorder._record_invocation(
            agent_type=self._agent_type,
            model_name=self._model_name,
            cwd=self._cwd,
            kwargs=self._kwargs,
            prompt=text,
            turn_index=self._turn_index,
        )
        rule = self._recorder._resolve_rule(inv)
        if rule.entered_event is not None:
            rule.entered_event.set()
        if rule.barrier is not None:
            rule.barrier.wait(timeout=5.0)
        if rule.block_event is not None:
            while not rule.block_event.is_set():
                if self._cancel_event is not None and self._cancel_event.is_set():
                    self.cancelled = True
                    return PromptResult(stop_reason="cancelled", text="", output_tokens=0)
                rule.block_event.wait(timeout=0.02)
        if rule.cancel_wait_s > 0:
            if self._cancel_event is not None:
                if self._cancel_event.wait(timeout=rule.cancel_wait_s):
                    self.cancelled = True
                    return PromptResult(stop_reason="cancelled", text="", output_tokens=0)
            else:
                threading.Event().wait(timeout=rule.cancel_wait_s)

        for ev in rule.events:
            if on_event is not None:
                on_event(ev)

        if rule.raise_on_prompt is not None:
            raise rule.raise_on_prompt

        out = rule.output(text, self._turn_index, inv) if callable(rule.output) else rule.output
        return PromptResult(
            stop_reason=rule.stop_reason,
            text=str(out),
            output_tokens=rule.output_tokens,
        )

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool:
        del wait, timeout
        self.cancelled = True
        self._recorder.cancelled_sessions += 1
        return True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._recorder.closed_sessions += 1


class WorkflowSessionRecorder:
    """Thread-safe factory + invocation recorder for ``create_engine_session``."""

    def __init__(
        self,
        resolver: Callable[[SessionInvocation], ScriptedRule | str | dict[str, Any]] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._session_seq = 0
        self._resolver = resolver
        self._start_rules: list[tuple[Callable[[str, str | None], bool], BaseException]] = []
        self.factory_calls: list[dict[str, Any]] = []
        self.invocations: list[SessionInvocation] = []
        self.closed_sessions = 0
        self.cancelled_sessions = 0

    def next_session_num(self) -> int:
        with self._lock:
            self._session_seq += 1
            return self._session_seq

    def fail_on_start(
        self,
        predicate: Callable[[str, str | None], bool],
        exc: BaseException,
    ) -> None:
        with self._lock:
            self._start_rules.append((predicate, exc))

    def _record_invocation(
        self,
        *,
        agent_type: str,
        model_name: str | None,
        cwd: str,
        kwargs: dict[str, Any],
        prompt: str,
        turn_index: int,
    ) -> SessionInvocation:
        with self._lock:
            self._seq += 1
            inv = SessionInvocation(
                sequence=self._seq,
                agent_type=agent_type,
                model_name=model_name,
                cwd=cwd,
                kwargs=dict(kwargs),
                prompt=prompt,
                turn_index=turn_index,
                thread_name=threading.current_thread().name,
            )
            self.invocations.append(inv)
            return inv

    def _resolve_rule(self, inv: SessionInvocation) -> ScriptedRule:
        if self._resolver is None:
            return ScriptedRule(output=json.dumps({"summary": "ok", "agent": inv.agent_type}))
        resolved = self._resolver(inv)
        if isinstance(resolved, ScriptedRule):
            return resolved
        if isinstance(resolved, dict):
            return ScriptedRule(output=json.dumps(resolved, ensure_ascii=False))
        return ScriptedRule(output=str(resolved))

    def factory(
        self,
        agent_type: str,
        cwd: str,
        on_rate_limit: Callable[[int], None] | None = None,
        cancel_event: threading.Event | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> ScriptedAgentSession:
        del on_rate_limit
        with self._lock:
            self.factory_calls.append(
                {
                    "agent_type": agent_type,
                    "cwd": cwd,
                    "model_name": model_name,
                    "kwargs": dict(kwargs),
                }
            )
            start_rules = list(self._start_rules)
        for pred, exc in start_rules:
            if pred(agent_type, model_name):
                raise exc
        return ScriptedAgentSession(
            self,
            agent_type=agent_type,
            cwd=cwd,
            model_name=model_name,
            cancel_event=cancel_event,
            kwargs=kwargs,
        )

    def patch(self, monkeypatch: Any) -> "WorkflowSessionRecorder":
        monkeypatch.setattr("src.agent_session.factory.create_engine_session", self.factory)
        monkeypatch.setattr("src.agent_session.create_engine_session", self.factory)
        return self


class FakePopen:
    """Minimal ``subprocess.Popen`` stand-in for CLI backend wire verification."""

    def __init__(self, cmd: list[str], **kwargs: Any) -> None:
        self.cmd = list(cmd)
        self.kwargs = dict(kwargs)
        self.pid = 42001
        self.returncode: int | None = None
        self.stdin = MagicMock()
        self.stdout = iter([json.dumps({"type": "result", "result": "ok", "stop_reason": "end_turn"}) + "\n"])
        self.stderr = iter([])

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class FakeACPClientSession:
    """Fake ACP wire session recording ``set_config_option`` and session rebuilds."""

    def __init__(self, *, agent_type: str = "codex", cwd: str = ".", model_name: str | None = None) -> None:
        self._agent_type = agent_type
        self._cwd = cwd
        self._model_name = model_name
        self._agent_cmd = agent_type
        self._agent_args = ["--acp"]
        self._capture_full_tool_content = True
        self.config_calls: list[tuple[str, str]] = []
        self.started = False
        self.closed = False

    async def set_config_option(self, config_id: str, value: str) -> bool:
        self.config_calls.append((config_id, value))
        return True

    def start(self, startup_timeout: float = 20.0) -> str:
        del startup_timeout
        self.started = True
        return "acp-fake-sess"

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class IMDeliveryRecord:
    """One IM delivery observed on ``FakeFeishuIMClient``."""

    op: str
    target_id: str
    msg_type: str
    content: str
    parsed: dict[str, Any] | None
    message_id: str
    file_key: str | None = None
    reply_in_thread: bool = False


class FakeFeishuIMClient:
    """Transport-boundary Feishu IM recorder supporting cards, patches, and files."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self.deliveries: list[IMDeliveryRecord] = []
        self.cards_by_msg_id: dict[str, list[dict[str, Any]]] = {}
        self.uploaded_files: dict[str, dict[str, Any]] = {}
        self.file_replies: list[dict[str, Any]] = []
        self.reactions: list[tuple[str, str]] = []
        self.fail_patch = False
        self.fail_upload = False
        self.fail_reply_file = False

    def _next_msg_id(self, prefix: str) -> str:
        self._seq += 1
        return f"om_{prefix}_{self._seq}"

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | None:
        try:
            loaded = json.loads(content)
            return loaded if isinstance(loaded, dict) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def reply_message(
        self,
        message_id: str,
        content: str,
        msg_type: str = "text",
        reply_in_thread: bool = False,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        with self._lock:
            out_id = self._next_msg_id("reply")
            parsed = self._parse_json(content) if msg_type == "interactive" else None
            self.deliveries.append(
                IMDeliveryRecord(
                    op="reply",
                    target_id=message_id,
                    msg_type=msg_type,
                    content=content,
                    parsed=parsed,
                    message_id=out_id,
                    reply_in_thread=bool(reply_in_thread),
                )
            )
            if parsed is not None:
                self.cards_by_msg_id.setdefault(out_id, []).append(parsed)
        return SimpleNamespace(success=lambda: True, code=0, msg="ok", data=SimpleNamespace(message_id=out_id))

    def send_message(
        self,
        receive_id_type: str,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        **_kwargs: Any,
    ) -> SimpleNamespace:
        del receive_id_type
        with self._lock:
            out_id = self._next_msg_id("send")
            parsed = self._parse_json(content) if msg_type == "interactive" else None
            self.deliveries.append(
                IMDeliveryRecord(
                    op="send",
                    target_id=receive_id,
                    msg_type=msg_type,
                    content=content,
                    parsed=parsed,
                    message_id=out_id,
                )
            )
            if parsed is not None:
                self.cards_by_msg_id.setdefault(out_id, []).append(parsed)
        return SimpleNamespace(success=lambda: True, code=0, msg="ok", data=SimpleNamespace(message_id=out_id))

    def patch_message(self, message_id: str, content: str, **_kwargs: Any) -> SimpleNamespace:
        if self.fail_patch:
            return SimpleNamespace(success=lambda: False, code=500, msg="patch failed")
        with self._lock:
            if message_id not in self.cards_by_msg_id:
                return SimpleNamespace(
                    success=lambda: False,
                    code=230001,
                    msg=f"cannot patch non-card or unknown message_id: {message_id}",
                )
            parsed = self._parse_json(content)
            self.deliveries.append(
                IMDeliveryRecord(
                    op="patch",
                    target_id=message_id,
                    msg_type="interactive",
                    content=content,
                    parsed=parsed,
                    message_id=message_id,
                )
            )
            if parsed is not None:
                self.cards_by_msg_id.setdefault(message_id, []).append(parsed)
        return SimpleNamespace(success=lambda: True, code=0, msg="ok")

    def upload_file(self, file_path: str, file_type: str = "stream", file_name: str | None = None) -> str | None:
        if self.fail_upload:
            return None
        path = Path(file_path)
        data = path.read_bytes() if path.exists() else b""
        with self._lock:
            file_key = f"file_key_{len(self.uploaded_files) + 1}"
            self.uploaded_files[file_key] = {
                "file_path": str(file_path),
                "file_type": file_type,
                "file_name": file_name or path.name,
                "bytes": data,
                "text": data.decode("utf-8", errors="replace"),
            }
        return file_key

    def reply_file(
        self,
        message_id: str,
        file_key: str,
        reply_in_thread: bool = False,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        if self.fail_reply_file:
            return SimpleNamespace(success=lambda: False, code=500, msg="reply_file failed")
        with self._lock:
            out_id = self._next_msg_id("file")
            self.file_replies.append(
                {"target_id": message_id, "file_key": file_key, "reply_in_thread": reply_in_thread}
            )
            self.deliveries.append(
                IMDeliveryRecord(
                    op="reply_file",
                    target_id=message_id,
                    msg_type="file",
                    content=file_key,
                    parsed=None,
                    message_id=out_id,
                    file_key=file_key,
                    reply_in_thread=bool(reply_in_thread),
                )
            )
        return SimpleNamespace(success=lambda: True, code=0, msg="ok", data=SimpleNamespace(message_id=out_id))

    def add_reaction(self, message_id: str, emoji_type: str) -> SimpleNamespace:
        with self._lock:
            self.reactions.append((message_id, emoji_type))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(reaction_id=f"r_{len(self.reactions)}"))

    def delete_reaction(self, *_a: Any, **_kw: Any) -> SimpleNamespace:
        return SimpleNamespace(success=lambda: True)

    def all_cards(self) -> list[dict[str, Any]]:
        with self._lock:
            return [rec.parsed for rec in self.deliveries if rec.parsed is not None]

    def latest_card(self, message_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if message_id is not None:
                history = self.cards_by_msg_id.get(message_id) or []
                return history[-1] if history else None
            cards = [rec.parsed for rec in self.deliveries if rec.parsed is not None]
            return cards[-1] if cards else None

    def latest_card_message_id(self) -> str | None:
        with self._lock:
            for rec in reversed(self.deliveries):
                if rec.parsed is not None:
                    return rec.message_id
            return None


class FakeCOTHttpClient:
    """Recorded transport client passed via ``api_client_factory()`` to ``FeishuCOTAPIClient``."""

    def __init__(self, *, fail_on_create: bool = False, fail_on_update: bool = False) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self.fail_on_create = fail_on_create
        self.fail_on_update = fail_on_update
        self.requests: list[Any] = []
        self.events: list[dict[str, Any]] = []

    def request(self, request: Any) -> BaseResponse:
        with self._lock:
            self.requests.append(request)
            body = getattr(request, "body", None)
            uri = str(getattr(request, "uri", "") or "")
            raw_events = body.get("events", []) if isinstance(body, dict) else []
            for ev in raw_events:
                if isinstance(ev, dict):
                    self.events.append(dict(ev))
            http_method = str(getattr(getattr(request, "http_method", None), "name", getattr(request, "http_method", ""))).upper()
            is_create = http_method == "POST" and "complete" not in uri and isinstance(body, dict) and "cot_id" not in body
            if is_create and self.fail_on_create:
                return self._build_response({}, status_code=500, code=500, msg="cot create failed")
            if not is_create and self.fail_on_update:
                return self._build_response({}, status_code=500, code=500, msg="cot update failed")
            if is_create:
                self._seq += 1
                return self._build_response(
                    {"cot_id": f"cot_e2e_{self._seq}", "message_id": f"om_cot_e2e_{self._seq}"}
                )
            return self._build_response({})

    @staticmethod
    def _build_response(data: dict[str, Any], *, status_code: int = 200, code: int = 0, msg: str = "ok") -> BaseResponse:
        resp = BaseResponse()
        resp.code = code
        resp.raw = RawResponse()
        resp.raw.status_code = status_code
        resp.raw.content = json.dumps({"code": code, "msg": msg, "data": data}, ensure_ascii=False).encode("utf-8")
        return resp


def collect_card_buttons(node: object) -> list[dict[str, Any]]:
    """Return every enabled non-link ``button`` element inside a Card 2.0 tree."""
    buttons: list[dict[str, Any]] = []
    stack: list[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if cur.get("tag") == "button":
                behaviors = cur.get("behaviors")
                is_open_url = isinstance(behaviors, list) and any(
                    isinstance(b, dict) and b.get("type") == "open_url" for b in behaviors
                )
                if not cur.get("disabled") and "multi_url" not in cur and not is_open_url:
                    buttons.append(cur)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return buttons


def collect_card_selects(node: object) -> list[dict[str, Any]]:
    """Return every ``select_static`` element inside a Card 2.0 tree."""
    selects: list[dict[str, Any]] = []
    stack: list[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if cur.get("tag") == "select_static":
                selects.append(cur)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return selects


def find_card_select(
    card: object,
    *,
    option_values: tuple[str, ...] = (),
    name_tokens: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Find a Card 2.0 ``select_static`` by option value or element name/placeholder."""
    for sel in collect_card_selects(card):
        opts = {
            str(opt.get("value", "")).lower()
            for opt in (sel.get("options") or [])
            if isinstance(opt, dict)
        }
        if option_values and any(cand.lower() in opts for cand in option_values):
            return sel
        name_blob = " ".join(
            [
                str(sel.get("name") or ""),
                str((sel.get("placeholder") or {}).get("content") or ""),
                json.dumps(sel.get("behaviors") or [], ensure_ascii=False),
            ]
        ).lower()
        if name_tokens and any(tok.lower() in name_blob for tok in name_tokens):
            return sel
    return None


def find_card_button(
    card: object,
    *,
    action_tokens: tuple[str, ...] = (),
    text_tokens: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Find a Card 2.0 callback ``button`` by action name or display text."""
    for btn in collect_card_buttons(card):
        behaviors = btn.get("behaviors") or []
        val = behaviors[0].get("value") if behaviors and isinstance(behaviors[0], dict) else {}
        action = str((val or {}).get("action") or "").lower()
        label = str((btn.get("text") or {}).get("content") or "").lower()
        if action_tokens and any(tok.lower() in action for tok in action_tokens):
            return btn
        if text_tokens and any(tok.lower() in label for tok in text_tokens):
            return btn
    return None


def click_card_element(
    graph: "WorkflowE2EGraph",
    *,
    message_id: str,
    element: dict[str, Any],
    option: str | None = None,
    chat_id: str = "oc_e2e",
    operator_id: str = "ou_e2e",
) -> tuple[Any, int, dict[str, Any] | None]:
    """Click a Card 2.0 button/select and return ``(resp, submitted, updated_card)``."""
    behaviors = element.get("behaviors") or []
    value = dict(behaviors[0].get("value") or {}) if behaviors and isinstance(behaviors[0], dict) else {}
    tag = str(element.get("tag") or "button")
    resp, submitted = graph.click_card(
        message_id=message_id,
        value=value,
        chat_id=chat_id,
        operator_id=operator_id,
        option=option,
        tag=tag,
    )
    updated: dict[str, Any] | None = None
    if resp is not None and getattr(resp, "card", None) is not None:
        raw_data = getattr(resp.card, "data", None)
        if isinstance(raw_data, dict):
            assert_card_2_invariants(raw_data, label=f"callback-card-{tag}")
            updated = raw_data
    if updated is None:
        updated = graph.fake_im.latest_card(message_id)
    return resp, submitted, updated


def extract_card_text(node: object) -> str:
    """Flatten all string leaves inside a card JSON payload for assertion checks."""
    parts: list[str] = []
    stack: list[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            parts.append(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return "\n".join(parts)


def assert_card_2_invariants(card: dict[str, Any], *, label: str = "card") -> None:
    """Strictly verify Feishu Card JSON 2.0 structure, budgets, and behaviors."""
    require_card_json_2(card)
    payload_bytes = len(json.dumps(card, ensure_ascii=False).encode("utf-8"))
    assert payload_bytes <= 27 * 1024, f"{label}: byte size {payload_bytes} > 27KB"
    nodes = count_tagged_nodes(card)
    assert nodes <= 180, f"{label}: tagged nodes {nodes} > 180"
    table_count = _count_explicit_table_nodes(card) + _count_markdown_tables_in_payload(card)
    assert table_count <= 5, f"{label}: table count {table_count} > 5"

    summary = ((card.get("config") or {}).get("summary") or {}).get("content")
    if isinstance(summary, str):
        assert 8 <= len(summary) <= 60, f"{label}: config.summary len {len(summary)} not in 8..60"

    for btn in collect_card_buttons(card):
        behaviors = btn.get("behaviors")
        assert isinstance(behaviors, list) and behaviors, f"{label}: button missing behaviors: {btn}"
        assert any(isinstance(b, dict) and b.get("type") == "callback" for b in behaviors), (
            f"{label}: button missing callback behavior: {btn}"
        )
        if "value" in btn:
            assert behaviors == [{"type": "callback", "value": btn["value"]}], (
                f"{label}: button behaviors mismatch value: {btn}"
            )

    for sel in collect_card_selects(card):
        assert "width" not in sel, f"{label}: select_static must not contain 'width': {sel}"
        behaviors = sel.get("behaviors")
        assert isinstance(behaviors, list) and behaviors, f"{label}: select_static missing behaviors: {sel}"
        assert any(isinstance(b, dict) and b.get("type") == "callback" for b in behaviors), (
            f"{label}: select_static missing callback behavior: {sel}"
        )
        if "value" in sel:
            assert behaviors == [{"type": "callback", "value": sel["value"]}], (
                f"{label}: select_static behaviors mismatch value: {sel}"
            )


def make_ws_message_event(
    text: str,
    *,
    idx: int = 1,
    message_id: str | None = None,
    chat_id: str = "oc_e2e",
    chat_type: str = "p2p",
    sender_id: str = "ou_e2e",
    root_id: str | None = None,
    parent_id: str | None = None,
) -> SimpleNamespace:
    """Build a realistic ``im.message.receive_v1`` event payload."""
    return SimpleNamespace(
        header=SimpleNamespace(tenant_key="tenant_e2e", event_id=f"ev_msg_{idx}_{sender_id}"),
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(open_id=sender_id, union_id=f"on_{sender_id}", user_id=f"u_{sender_id}"),
                sender_type="user",
            ),
            message=SimpleNamespace(
                message_id=message_id or f"om_e2e_{idx}",
                chat_id=chat_id,
                chat_type=chat_type,
                message_type="text",
                content=json.dumps({"text": text}, ensure_ascii=False),
                create_time=str(int(time.time() * 1000)),
                mentions=[],
                root_id=root_id,
                parent_id=parent_id,
                thread_id=root_id,
            ),
        ),
    )


def drive_card_callback(
    client: FeishuWSClient,
    *,
    message_id: str,
    chat_id: str = "oc_e2e",
    operator_id: str = "ou_e2e",
    value: dict[str, Any] | None = None,
    behaviors: list[dict[str, Any]] | None = None,
    option: str | None = None,
    options: list[str] | None = None,
    form_value: dict[str, Any] | None = None,
    input_value: str | None = None,
    event_id: str | None = None,
    tag: str = "button",
    name: str = "",
) -> tuple[Any, int]:
    """Send a real Card 2.0 ``card.action.trigger`` event through ``FeishuWSClient``."""
    resolved_behaviors = (
        behaviors
        if behaviors is not None
        else ([{"type": "callback", "value": value}] if value is not None else None)
    )
    trigger = SimpleNamespace(
        schema="2.0",
        header=SimpleNamespace(
            event_id=event_id or f"ev_card_{time.time_ns()}",
            event_type="card.action.trigger",
            tenant_key="tenant_e2e",
        ),
        event=SimpleNamespace(
            operator=SimpleNamespace(open_id=operator_id, user_id=None, union_id=f"on_{operator_id}"),
            context=SimpleNamespace(open_message_id=message_id, open_chat_id=chat_id),
            action=SimpleNamespace(
                tag=tag,
                name=name,
                value=value if value is not None else {},
                behaviors=resolved_behaviors,
                option=option,
                options=options,
                form_value=form_value,
                input_value=input_value,
            ),
        ),
    )
    before = client._scheduler.submit.call_count
    resp = client._handle_card_action_callback(trigger)
    client._finish_card_advisories(True)
    submitted = client._scheduler.submit.call_count - before
    return resp, submitted


class WorkflowE2EGraph:
    """Wired real FeishuWSClient graph + fake transport boundaries."""

    def __init__(
        self,
        *,
        client: FeishuWSClient,
        project: Any,
        fake_im: FakeFeishuIMClient,
        fake_cot: FakeCOTHttpClient,
        settings: Settings,
        repo_dir: Path,
        projects_config_file: Path,
    ) -> None:
        self.client = client
        self.project = project
        self.fake_im = fake_im
        self.fake_cot = fake_cot
        self.settings = settings
        self.repo_dir = repo_dir
        self.projects_config_file = projects_config_file
        self._msg_seq = 0
        self._worker_threads: list[threading.Thread] = []

    def send_text(
        self,
        text: str,
        *,
        chat_id: str = "oc_e2e",
        sender_id: str = "ou_e2e",
        chat_type: str = "p2p",
        root_id: str | None = None,
        parent_id: str | None = None,
    ) -> str:
        self._msg_seq += 1
        msg_id = f"om_user_{self._msg_seq}"
        event = make_ws_message_event(
            text,
            idx=self._msg_seq,
            message_id=msg_id,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_id=sender_id,
            root_id=root_id,
            parent_id=parent_id,
        )
        self.client._handle_message(event)
        return msg_id

    def click_card(
        self,
        *,
        message_id: str,
        value: dict[str, Any],
        chat_id: str = "oc_e2e",
        operator_id: str = "ou_e2e",
        option: str | None = None,
        tag: str = "button",
    ) -> tuple[Any, int]:
        return drive_card_callback(
            self.client,
            message_id=message_id,
            chat_id=chat_id,
            operator_id=operator_id,
            value=value,
            option=option,
            tag=tag,
        )

    def reload_project_manager(self):
        return self.client._project_manager.__class__(str(self.projects_config_file))

    def close(self) -> None:
        for t in self._worker_threads:
            t.join(timeout=2.0)
        try:
            self.client.close()
        except Exception:
            pass


def build_workflow_e2e_graph(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    chat_id: str = "oc_e2e",
    cot_enabled: bool = False,
    threaded_scheduler: bool = False,
    extra_settings: dict[str, Any] | None = None,
) -> WorkflowE2EGraph:
    """Build real FeishuWSClient + MessageDispatcher + WorkflowHandler + ProjectManager."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    projects_config_file = fake_home / ".ghostap" / "projects.json"

    settings_kwargs: dict[str, Any] = {
        "app_id": "cli_wf_e2e",
        "app_secret": "secret_wf_e2e",
        "admin_user_ids": "ou_e2e",
        "allowed_user_ids": "ou_e2e",
        "default_working_dir": str(repo_dir),
        "projects_dir": str(tmp_path / "projects"),
        "projects_config_file": str(projects_config_file),
        "feishu_cot_enabled": cot_enabled,
        "streaming_enabled": False,
        "ingress_access_mode": "legacy_allow_all",
    }
    if extra_settings:
        settings_kwargs.update(extra_settings)
    settings = Settings(**settings_kwargs)

    fake_im = FakeFeishuIMClient()
    fake_cot = FakeCOTHttpClient()
    run_counter = [0]
    worker_threads: list[threading.Thread] = []

    def _submit_task(spec: Any, callback: Callable[[Any], Any]) -> SimpleNamespace:
        run_counter[0] += 1
        run_id = f"run_wf_e2e_{run_counter[0]}"
        task_ctx = SimpleNamespace(run_id=run_id, spec=spec)
        handle = SimpleNamespace(
            run_id=run_id,
            status=TaskStatus.SUCCEEDED,
            error=None,
            add_done_callback=lambda _fn: None,
        )
        if threaded_scheduler:
            t = threading.Thread(target=callback, args=(task_ctx,), daemon=True, name=f"wf-e2e-{run_id}")
            worker_threads.append(t)
            t.start()
        else:
            callback(task_ctx)
        return handle

    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()
    scheduler.list_tasks = MagicMock(return_value=[])
    scheduler.get_state = MagicMock(return_value=None)
    scheduler.update_project_id = MagicMock()
    scheduler.submit = MagicMock(side_effect=_submit_task)

    with (
        patch("src.feishu.ws_client.get_settings", return_value=settings),
        patch("src.config.get_settings", return_value=settings),
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
    ):
        client = FeishuWSClient(lambda *_a: None)

    client._get_api_client = lambda: fake_cot  # type: ignore[method-assign]
    client._handler_ctx._im_client = fake_im
    client._handler_ctx.api_client_factory = lambda: fake_cot
    for handler in client._handler_ctx.handlers.values():
        handler._im_client = fake_im
        if hasattr(handler, "im_client"):
            try:
                handler.im_client = fake_im
            except AttributeError:
                pass

    client._project_manager.create_project("demo", "DemoProj", str(repo_dir), chat_id=chat_id)
    project = client._project_manager.get_project_for_chat("demo", chat_id)
    assert project is not None
    client._project_manager.set_active_project(chat_id, project.project_id)

    graph = WorkflowE2EGraph(
        client=client,
        project=project,
        fake_im=fake_im,
        fake_cot=fake_cot,
        settings=settings,
        repo_dir=repo_dir,
        projects_config_file=projects_config_file,
    )
    graph._worker_threads = worker_threads
    return graph


@dataclass
class WorkflowE2EResult:
    """Outputs captured from a real ``WorkflowEngine`` + Node.js execution."""

    project: WorkflowProject
    engine: WorkflowEngine
    progress_snapshots: list[WorkflowProject] = field(default_factory=list)
    agent_starts: list[tuple[str, str]] = field(default_factory=list)
    agent_done_payloads: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    report_files: WorkflowReportFiles | None = None
    recorder: WorkflowSessionRecorder | None = None


def run_workflow_e2e(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    script_content: str,
    run_spec: WorkflowRunSpec,
    recorder: WorkflowSessionRecorder | None = None,
    chat_id: str = "oc_e2e",
    generate_reports: bool = False,
    coalescer_debounce_s: float = 0.02,
    short_timeout_floor_s: int | None = None,
    fast_retries: bool = True,
    extra_callbacks: dict[str, Any] | None = None,
) -> WorkflowE2EResult:
    """Execute real ``WorkflowEngine`` + Node.js ``runtime.js`` + ``AgentExecutor``."""
    if recorder is not None:
        recorder.patch(monkeypatch)
    if fast_retries:
        monkeypatch.setattr("src.workflow_engine.executor.AgentExecutor._sleep_with_backoff", lambda *_a, **_kw: None)
    if short_timeout_floor_s is not None:
        monkeypatch.setattr(
            "src.workflow_engine.bridge._settings_int",
            lambda _attr, default: short_timeout_floor_s if "call_timeout" in _attr else default,
        )
        monkeypatch.setattr(
            "src.workflow_engine.executor._settings_int",
            lambda _attr, default: short_timeout_floor_s if "call_timeout" in _attr else default,
        )

    script_path = tmp_path / f"e2e_workflow_{time.time_ns()}.js"
    script_path.write_text(script_content, encoding="utf-8")

    progress_snapshots: list[WorkflowProject] = []
    agent_starts: list[tuple[str, str]] = []
    agent_done_payloads: list[tuple[str, dict[str, Any]]] = []
    coalescer = ProgressCoalescer(
        on_progress=lambda snap: progress_snapshots.append(snap),
        debounce_s=coalescer_debounce_s,
    )

    cb_kwargs: dict[str, Any] = {
        "on_progress": lambda snap: coalescer.enqueue(snap),
        "on_agent_start": lambda label, tool: agent_starts.append((label, tool)),
        "on_agent_done": lambda label, payload: agent_done_payloads.append((label, dict(payload))),
    }
    if extra_callbacks:
        cb_kwargs.update(extra_callbacks)
    callbacks = WorkflowEngineCallbacks(**cb_kwargs)

    engine = WorkflowEngine(chat_id=chat_id, root_path=str(tmp_path))
    try:
        project = engine.execute_workflow(str(script_path), callbacks=callbacks, run_spec=run_spec)
    finally:
        coalescer.stop()

    report_files = write_workflow_report_files(project, root_path=str(tmp_path)) if generate_reports else None
    return WorkflowE2EResult(
        project=project,
        engine=engine,
        progress_snapshots=progress_snapshots,
        agent_starts=agent_starts,
        agent_done_payloads=agent_done_payloads,
        report_files=report_files,
        recorder=recorder,
    )


def make_run_spec(
    *,
    pool: Iterable[Any],
    task: str = "E2E workflow task",
    chat_id: str = "oc_e2e",
    orchestrator_agent_id: str | None = None,
    budget: int = 200,
    deadline: float | None = None,
    initiator_user_id: str = "ou_e2e",
) -> WorkflowRunSpec:
    """Convenience constructor for frozen ``WorkflowRunSpec`` in E2E tests."""
    from src.workflow_engine.agent_pool import freeze_agent_pool, select_auto_orchestrator
    from src.workflow_engine.run_spec import _review_binding

    frozen = freeze_agent_pool(pool)
    orch = (
        next((b for b in frozen if b.agent_id == orchestrator_agent_id), None)
        if orchestrator_agent_id
        else select_auto_orchestrator(frozen)
    )
    assert orch is not None
    return WorkflowRunSpec(
        orchestrator=_review_binding(orch),
        reviewers=(),
        tool_model_map={},
        task=task,
        chat_id=chat_id,
        topic_id=None,
        budget=budget,
        deadline=deadline,
        auto_reviewer=True,
        initiator_user_id=initiator_user_id,
        allowed_tools=tuple(dict.fromkeys(b.tool_name for b in frozen)),
        agent_pool=frozen,
        orchestrator_agent_id=orch.agent_id,
    )
