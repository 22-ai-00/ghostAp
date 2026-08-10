"""Strict Codex rollout fallback for collaboration terminal snapshots.

The official Codex ACP adapter currently exposes ``agentsStates`` from the
app-server item, but does not forward the authoritative ``list_agents``
function output. This module reads only the active session's bounded rollout
tail and converts a fully attributable list result into canonical ACP child
state evidence. Any identity, time-window, or shape ambiguity fails closed.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .collaboration import merge_tool_call_sequence
from .models import PromptResult, ToolCallInfo

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9-]{16,128}")
_MAX_HEAD_BYTES = 1024 * 1024
_MAX_TAIL_BYTES = 4 * 1024 * 1024
_MAX_FUNCTION_OUTPUT_BYTES = 1024 * 1024
_ROLLOUT_SETTLE_TIMEOUT_S = 0.2
_ROLLOUT_SETTLE_INTERVAL_S = 0.02
CODEX_CHILD_LIFECYCLE_POLL_INTERVAL_S = 0.1
_CHILD_TURN_BOUNDARY_SLACK_S = 1.0
_MAX_TRACKED_CHILDREN = 200
_ROOT_AGENT_PATH = "/root"
_DIRECT_AGENT_PATH_RE = re.compile(
    r"/root/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
)
_AGENT_PATH_RE = re.compile(
    r"/root(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}){1,8}"
)
_SUBAGENT_ACTIVITY_KINDS = frozenset(
    {"started", "interacted", "interrupted"}
)
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_STATUS_MAP = {
    "pending": "pending",
    "pendinginit": "pending",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "errored": "failed",
    "notfound": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "interrupted": "cancelled",
    "shutdown": "cancelled",
}


def _event_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, OverflowError):
        return None


def _read_bounded_lines(
    path: Path,
    *,
    tail: bool,
) -> tuple[list[bytes], bool]:
    limit = _MAX_TAIL_BYTES if tail else _MAX_HEAD_BYTES
    try:
        with path.open("rb") as handle:
            if tail:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                start = max(0, size - limit)
                handle.seek(start)
                data = handle.read(limit)
                if start:
                    _, separator, data = data.partition(b"\n")
                    if not separator:
                        return [], False
            else:
                data = handle.read(limit)
    except OSError:
        return [], False
    complete = not tail or not data or data.endswith(b"\n")
    return data.splitlines(), complete


def _decode_line(raw_line: bytes) -> Mapping[str, Any] | None:
    try:
        value = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _session_meta_payload(path: Path) -> Mapping[str, Any] | None:
    lines, _ = _read_bounded_lines(path, tail=False)
    for raw_line in lines:
        event = _decode_line(raw_line)
        if event is None or event.get("type") != "session_meta":
            continue
        payload = event.get("payload")
        return payload if isinstance(payload, Mapping) else None
    return None


def _session_matches(path: Path, session_id: str, cwd: str) -> bool:
    payload = _session_meta_payload(path)
    if payload is None:
        return False
    expected_cwd = os.path.realpath(cwd)
    raw_ids = (payload.get("id"), payload.get("session_id"))
    raw_cwd = payload.get("cwd")
    return (
        any(
            isinstance(raw_id, str) and raw_id == session_id
            for raw_id in raw_ids
        )
        and isinstance(raw_cwd, str)
        and os.path.realpath(raw_cwd) == expected_cwd
    )


def _rollout_candidates(
    *,
    session_id: str,
    codex_home: str | None,
) -> set[Path]:
    if _SESSION_ID_RE.fullmatch(session_id) is None:
        return set()
    home = Path(
        codex_home
        or os.environ.get("CODEX_HOME")
        or (Path.home() / ".codex")
    ).expanduser()
    root = home / "sessions"
    try:
        resolved_root = root.resolve(strict=True)
        candidates = list(root.rglob(f"rollout-*-{session_id}.jsonl"))
    except OSError:
        return set()
    matches: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if candidate.is_symlink() or not resolved.is_relative_to(
                resolved_root
            ):
                continue
        except OSError:
            continue
        matches.add(resolved)
    return matches


def _find_rollout(
    *,
    session_id: str,
    cwd: str,
    codex_home: str | None,
) -> Path | None:
    matches = {
        candidate
        for candidate in _rollout_candidates(
            session_id=session_id,
            codex_home=codex_home,
        )
        if _session_matches(candidate, session_id, cwd)
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _normalize_agent_status(value: object) -> str | None:
    raw_status: object = value
    if isinstance(value, Mapping):
        if len(value) != 1:
            return None
        raw_status = next(iter(value))
    if not isinstance(raw_status, str):
        return None
    return _STATUS_MAP.get(raw_status.strip().casefold())


def _latest_list_agents_output(
    path: Path,
    *,
    started_at: float,
    ended_at: float,
) -> tuple[str, list[Mapping[str, Any]]] | None:
    if ended_at < started_at:
        return None
    calls: list[tuple[float, int, object, object]] = []
    outputs: list[tuple[float, int, str, object]] = []
    lines, complete = _read_bounded_lines(path, tail=True)
    if not complete:
        return None
    for line_index, raw_line in enumerate(lines):
        event = _decode_line(raw_line)
        if event is None:
            return None
        if event.get("type") != "response_item":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        payload_type = payload.get("type")
        if (
            payload_type == "function_call"
            and payload.get("name") == "list_agents"
            and payload.get("namespace") == "collaboration"
        ):
            timestamp = _event_timestamp(event.get("timestamp"))
            if timestamp is None:
                return None
            if started_at <= timestamp <= ended_at:
                calls.append(
                    (
                        timestamp,
                        line_index,
                        payload.get("call_id"),
                        payload.get("arguments"),
                    )
                )
            continue
        if payload_type != "function_call_output":
            continue
        timestamp = _event_timestamp(event.get("timestamp"))
        if timestamp is None:
            continue
        if not started_at <= timestamp <= ended_at:
            continue
        call_id = payload.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        outputs.append(
            (timestamp, line_index, call_id.strip(), payload.get("output"))
        )

    if not calls:
        return None
    call_timestamp, call_index, raw_call_id, raw_arguments = calls[-1]
    if not isinstance(raw_call_id, str) or not raw_call_id.strip():
        return None
    if not isinstance(raw_arguments, str):
        return None
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, Mapping) or arguments:
        return None
    call_id = raw_call_id.strip()
    matching_outputs = [
        (timestamp, line_index, raw_output)
        for timestamp, line_index, output_call_id, raw_output in outputs
        if output_call_id == call_id
        and timestamp >= call_timestamp
        and line_index > call_index
    ]
    if len(matching_outputs) != 1:
        return None
    _, _, raw_output = matching_outputs[0]
    if (
        not isinstance(raw_output, str)
        or len(raw_output.encode("utf-8")) > _MAX_FUNCTION_OUTPUT_BYTES
    ):
        return None
    try:
        output = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    if not isinstance(output, Mapping):
        return None
    raw_agents = output.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        return None
    agents = [agent for agent in raw_agents if isinstance(agent, Mapping)]
    if len(agents) != len(raw_agents):
        return None
    return call_id, agents


def _result_child_sources(
    result: PromptResult,
) -> tuple[frozenset[str], bool]:
    """Collect stable child ids already represented by the ACP result."""
    sources: set[str] = set()
    malformed = False

    def add(raw_source: object) -> None:
        nonlocal malformed
        if not isinstance(raw_source, str):
            malformed = True
            return
        source = raw_source.strip()
        if not source:
            malformed = True
            return
        sources.add(source)

    for tool_call in result.tool_calls:
        raw_source = getattr(tool_call, "subagent_source_id", None)
        if raw_source not in (None, ""):
            add(raw_source)
        for raw_container in (
            getattr(tool_call, "collaboration_receivers", ()),
            getattr(tool_call, "subagent_states", ()),
        ):
            if raw_container is None:
                continue
            if isinstance(
                raw_container,
                (str, bytes, bytearray, Mapping),
            ) or not isinstance(raw_container, Iterable):
                malformed = True
                continue
            for item in raw_container:
                if isinstance(item, Mapping):
                    add(item.get("source_id"))
                else:
                    add(item)
    return frozenset(sources), malformed


def _parent_activity_paths(
    path: Path,
    *,
    target_sources: frozenset[str],
    started_at: float,
    ended_at: float,
) -> tuple[dict[str, str], bool, bool]:
    """Read unique direct-child identities from one bounded parent tail."""
    if ended_at < started_at:
        return {}, True, False
    source_to_path: dict[str, str] = {}
    path_to_source: dict[str, str] = {}
    saw_target = False
    lines, complete = _read_bounded_lines(path, tail=True)
    if not complete:
        return {}, False, True
    for raw_line in lines:
        event = _decode_line(raw_line)
        if event is None:
            return {}, False, True
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload")
        if (
            not isinstance(payload, Mapping)
            or payload.get("type") != "sub_agent_activity"
        ):
            continue
        timestamp = _event_timestamp(event.get("timestamp"))
        if timestamp is None:
            return {}, True, False
        if timestamp < started_at or timestamp > ended_at:
            continue
        raw_occurred_at_ms = payload.get("occurred_at_ms")
        if (
            not isinstance(raw_occurred_at_ms, int)
            or isinstance(raw_occurred_at_ms, bool)
            or abs(raw_occurred_at_ms / 1000.0 - timestamp) > 1.0
        ):
            return {}, True, False
        raw_source = payload.get("agent_thread_id")
        raw_agent_path = payload.get("agent_path")
        raw_kind = payload.get("kind")
        if (
            not isinstance(raw_source, str)
            or _SESSION_ID_RE.fullmatch(raw_source.strip()) is None
            or not isinstance(raw_agent_path, str)
            or _DIRECT_AGENT_PATH_RE.fullmatch(raw_agent_path.strip()) is None
            or not isinstance(raw_kind, str)
            or raw_kind.strip().casefold() not in _SUBAGENT_ACTIVITY_KINDS
        ):
            return {}, True, False
        source = raw_source.strip()
        agent_path = raw_agent_path.strip()
        if source not in target_sources:
            return {}, True, False
        saw_target = True
        if (
            source in source_to_path
            and source_to_path[source] != agent_path
        ) or (
            agent_path in path_to_source
            and path_to_source[agent_path] != source
        ):
            return {}, True, False
        source_to_path[source] = agent_path
        path_to_source[agent_path] = source
    if saw_target and set(source_to_path) != set(target_sources):
        return {}, True, False
    return source_to_path, False, False


def _path_to_source(result: PromptResult) -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    source_to_path: dict[str, str] = {}
    for tool_call in result.tool_calls:
        raw_path = getattr(tool_call, "subagent_path", None)
        raw_source = getattr(tool_call, "subagent_source_id", None)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        if not isinstance(raw_source, str) or not raw_source.strip():
            continue
        path = raw_path.strip()
        source = raw_source.strip()
        existing = mapping.get(path)
        if existing is not None and existing != source:
            return None
        existing_path = source_to_path.get(source)
        if existing_path is not None and existing_path != path:
            return None
        mapping[path] = source
        source_to_path[source] = path
    return mapping or None


def _is_subagent_session_meta(payload: Mapping[str, Any]) -> bool:
    thread_source = payload.get("thread_source")
    if isinstance(thread_source, str):
        return thread_source.strip().casefold() == "subagent"
    source = payload.get("source")
    if isinstance(source, str):
        return source.strip().casefold() == "subagent"
    if isinstance(source, Mapping):
        return "subagent" in source
    return False


def _child_identity_matches(
    path: Path,
    *,
    parent_session_id: str,
    child_thread_id: str,
    cwd: str,
    expected_agent_path: str,
    known_thread_paths: Mapping[str, str],
) -> bool:
    payload = _session_meta_payload(path)
    if payload is None:
        return False
    raw_cwd = payload.get("cwd")
    if (
        payload.get("id") != child_thread_id
        or payload.get("session_id") != parent_session_id
        or not isinstance(raw_cwd, str)
        or os.path.realpath(raw_cwd) != os.path.realpath(cwd)
        or not _is_subagent_session_meta(payload)
    ):
        return False
    parent_thread_id = payload.get("parent_thread_id")
    expected_parent_path = expected_agent_path.rpartition("/")[0]
    if (
        not isinstance(parent_thread_id, str)
        or known_thread_paths.get(parent_thread_id) != expected_parent_path
    ):
        return False
    recorded_path = payload.get("agent_path")
    return bool(
        isinstance(recorded_path, str)
        and recorded_path.strip() == expected_agent_path
    )


@dataclass(frozen=True)
class _ChildTurnObservation:
    generation: str
    status: str


def _latest_child_turn_observation(
    path: Path,
    *,
    logical_task_started_at: float,
    ended_at: float,
) -> _ChildTurnObservation | None:
    boundary = logical_task_started_at - _CHILD_TURN_BOUNDARY_SLACK_S
    active_generation = ""
    active_started_at: float | None = None
    active_status = ""
    active_has_turn_id = False
    lines, complete = _read_bounded_lines(path, tail=True)
    if not complete:
        return None
    for line_index, raw_line in enumerate(lines):
        event = _decode_line(raw_line)
        if event is None:
            return None
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        raw_type = payload.get("type")
        event_type = (
            raw_type.strip().casefold()
            if isinstance(raw_type, str)
            else ""
        )
        if event_type not in {
            "task_started",
            "turn_started",
            "task_complete",
            "turn_complete",
            "turn_aborted",
        }:
            continue
        timestamp = _event_timestamp(event.get("timestamp"))
        if timestamp is None:
            return None
        if timestamp < boundary or timestamp > ended_at:
            continue
        raw_turn_id = payload.get("turn_id")
        turn_id = (
            raw_turn_id.strip()
            if isinstance(raw_turn_id, str) and raw_turn_id.strip()
            else ""
        )
        if event_type in {"task_started", "turn_started"}:
            active_generation = turn_id or f"{timestamp:.6f}:{line_index}"
            active_started_at = timestamp
            active_status = "running"
            active_has_turn_id = bool(turn_id)
            continue
        if active_started_at is None or timestamp < active_started_at:
            continue
        if active_has_turn_id and not turn_id:
            continue
        if turn_id and turn_id != active_generation:
            continue
        active_status = (
            "cancelled"
            if event_type == "turn_aborted"
            else "completed"
        )
    if active_started_at is None or not active_generation or not active_status:
        return None
    return _ChildTurnObservation(active_generation, active_status)


class CodexChildLifecycleMonitor:
    """Project persisted child-turn terminals into canonical ACP evidence."""

    def __init__(
        self,
        *,
        parent_session_id: str,
        cwd: str,
        logical_task_started_at: float,
        codex_home: str | None = None,
    ) -> None:
        self._parent_session_id = parent_session_id
        self._cwd = cwd
        self._logical_task_started_at = logical_task_started_at
        self._codex_home = codex_home
        self._source_to_path: dict[str, str] = {}
        self._path_to_source: dict[str, str] = {}
        self._rollout_by_source: dict[str, Path] = {}
        self._invalid_sources: set[str] = set()
        self._next_lookup_at: dict[str, float] = {}
        self._last_fingerprint: tuple[tuple[str, str, str], ...] | None = None
        self._saw_rollout_candidate = False
        self._identity_ambiguous = False

    @property
    def saw_rollout_candidate(self) -> bool:
        return self._saw_rollout_candidate

    def observe_tool_call(self, tool_call: object) -> None:
        raw_source = getattr(tool_call, "subagent_source_id", None)
        raw_path = getattr(tool_call, "subagent_path", None)
        if raw_source in (None, "") and raw_path in (None, ""):
            return
        if (
            not isinstance(raw_source, str)
            or _SESSION_ID_RE.fullmatch(raw_source.strip()) is None
        ):
            self._identity_ambiguous = True
            return
        source = raw_source.strip()
        if raw_path in (None, ""):
            return
        if (
            not isinstance(raw_path, str)
            or _AGENT_PATH_RE.fullmatch(raw_path.strip()) is None
            or source == self._parent_session_id
        ):
            self._identity_ambiguous = True
            return
        agent_path = raw_path.strip()
        existing = self._source_to_path.get(source)
        if existing is not None and existing != agent_path:
            self._identity_ambiguous = True
            return
        existing_source = self._path_to_source.get(agent_path)
        if existing_source is not None and existing_source != source:
            self._identity_ambiguous = True
            return
        if (
            source not in self._source_to_path
            and len(self._source_to_path) >= _MAX_TRACKED_CHILDREN
        ):
            self._identity_ambiguous = True
            return
        self._source_to_path[source] = agent_path
        self._path_to_source[agent_path] = source

    def observe_result(self, result: PromptResult) -> None:
        for tool_call in result.tool_calls:
            self.observe_tool_call(tool_call)

    def _locate_rollouts(self) -> None:
        known_thread_paths = {
            self._parent_session_id: _ROOT_AGENT_PATH,
            **self._source_to_path,
        }
        now = time.monotonic()
        for source_id, agent_path in self._source_to_path.items():
            if (
                source_id in self._rollout_by_source
                or source_id in self._invalid_sources
                or now < self._next_lookup_at.get(source_id, 0.0)
            ):
                continue
            candidates = _rollout_candidates(
                session_id=source_id,
                codex_home=self._codex_home,
            )
            if not candidates:
                self._next_lookup_at[source_id] = now + 0.25
                continue
            self._saw_rollout_candidate = True
            if len(candidates) != 1:
                self._invalid_sources.add(source_id)
                continue
            candidate = next(iter(candidates))
            if not _child_identity_matches(
                candidate,
                parent_session_id=self._parent_session_id,
                child_thread_id=source_id,
                cwd=self._cwd,
                expected_agent_path=agent_path,
                known_thread_paths=known_thread_paths,
            ):
                self._invalid_sources.add(source_id)
                continue
            self._rollout_by_source[source_id] = candidate

    def poll(
        self,
        *,
        ended_at: float | None = None,
        require_all_terminal: bool = False,
        emit_unchanged: bool = False,
    ) -> ToolCallInfo | None:
        if self._identity_ambiguous or not self._source_to_path:
            return None
        self._locate_rollouts()
        if self._invalid_sources:
            return None
        observations: list[tuple[str, _ChildTurnObservation]] = []
        observed_until = time.time() if ended_at is None else ended_at
        for source_id in self._source_to_path:
            rollout = self._rollout_by_source.get(source_id)
            if rollout is None:
                continue
            observation = _latest_child_turn_observation(
                rollout,
                logical_task_started_at=self._logical_task_started_at,
                ended_at=observed_until,
            )
            if observation is not None:
                observations.append((source_id, observation))
        if not observations:
            return None
        if require_all_terminal and (
            len(observations) != len(self._source_to_path)
            or any(
                observation.status not in _TERMINAL_STATUSES
                for _, observation in observations
            )
        ):
            return None
        fingerprint = tuple(
            (source_id, observation.generation, observation.status)
            for source_id, observation in observations
        )
        if not emit_unchanged and fingerprint == self._last_fingerprint:
            return None
        self._last_fingerprint = fingerprint
        return ToolCallInfo(
            id=f"rollout-child-lifecycle:{self._parent_session_id}",
            title="list_agents",
            kind="other",
            status="completed",
            collaboration_tool="list_agents",
            collaboration_receivers=tuple(
                source_id for source_id, _ in observations
            ),
            subagent_states=tuple(
                {
                    "source_id": source_id,
                    "status": observation.status,
                    "message": "",
                }
                for source_id, observation in observations
            ),
        )


def _build_evidence(
    result: PromptResult,
    *,
    call_id: str,
    agents: list[Mapping[str, Any]],
) -> ToolCallInfo | None:
    path_to_source = _path_to_source(result)
    if path_to_source is None:
        return None
    observed: dict[str, str] = {}
    seen_names: set[str] = set()
    root_count = 0
    for agent in agents:
        raw_name = agent.get("agent_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None
        name = raw_name.strip()
        if name in seen_names:
            return None
        seen_names.add(name)
        status = _normalize_agent_status(agent.get("agent_status"))
        if status is None:
            return None
        if name == _ROOT_AGENT_PATH:
            root_count += 1
            continue
        if name in path_to_source:
            if status not in _TERMINAL_STATUSES:
                return None
            observed[name] = status
            continue
        return None
    if root_count != 1:
        return None
    if set(observed) != set(path_to_source):
        return None
    ordered_paths = tuple(path_to_source)
    return ToolCallInfo(
        id=f"rollout-list-agents:{call_id}",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        collaboration_receivers=tuple(
            path_to_source[path] for path in ordered_paths
        ),
        subagent_states=tuple(
            {
                "source_id": path_to_source[path],
                "status": observed[path],
                "message": "",
            }
            for path in ordered_paths
        ),
    )


def enrich_codex_reconciliation_result(
    result: PromptResult,
    *,
    session_id: str,
    cwd: str,
    logical_task_started_at: float | None = None,
    started_at: float,
    ended_at: float,
    codex_home: str | None = None,
) -> tuple[PromptResult, ToolCallInfo | None]:
    """Add authoritative child-turn evidence, then use parent fallback."""
    monitor = CodexChildLifecycleMonitor(
        parent_session_id=session_id,
        cwd=cwd,
        logical_task_started_at=(
            started_at
            if logical_task_started_at is None
            else logical_task_started_at
        ),
        codex_home=codex_home,
    )
    monitor.observe_result(result)
    rollout = _find_rollout(
        session_id=session_id,
        cwd=cwd,
        codex_home=codex_home,
    )
    if rollout is not None:
        target_sources, malformed_sources = _result_child_sources(result)
        if malformed_sources:
            return result, None
        identity_deadline = time.monotonic() + _ROLLOUT_SETTLE_TIMEOUT_S
        while True:
            (
                source_to_path,
                identity_ambiguous,
                retryable_identity_read,
            ) = _parent_activity_paths(
                rollout,
                target_sources=target_sources,
                started_at=(
                    started_at
                    if logical_task_started_at is None
                    else logical_task_started_at
                ),
                ended_at=ended_at,
            )
            if not retryable_identity_read:
                break
            if time.monotonic() >= identity_deadline:
                return result, None
            time.sleep(_ROLLOUT_SETTLE_INTERVAL_S)
        if identity_ambiguous:
            return result, None
        for source, agent_path in source_to_path.items():
            monitor.observe_tool_call(
                ToolCallInfo(
                    id=f"rollout-parent-activity:{source}",
                    title="sub_agent_activity",
                    kind="other",
                    status="completed",
                    subagent_source_id=source,
                    subagent_path=agent_path,
                    subagent_activity="interacted",
                )
            )
    settle_deadline = time.monotonic() + _ROLLOUT_SETTLE_TIMEOUT_S
    while True:
        child_evidence = monitor.poll(
            ended_at=ended_at,
            require_all_terminal=True,
            emit_unchanged=True,
        )
        if child_evidence is not None:
            return (
                replace(
                    result,
                    tool_calls=merge_tool_call_sequence(
                        [*result.tool_calls, child_evidence]
                    ),
                ),
                child_evidence,
            )
        if time.monotonic() >= settle_deadline:
            break
        time.sleep(_ROLLOUT_SETTLE_INTERVAL_S)
    if monitor.saw_rollout_candidate:
        return result, None
    if rollout is None:
        return result, None
    settle_deadline = time.monotonic() + _ROLLOUT_SETTLE_TIMEOUT_S
    latest = None
    while latest is None:
        latest = _latest_list_agents_output(
            rollout,
            started_at=started_at,
            ended_at=ended_at,
        )
        if latest is not None or time.monotonic() >= settle_deadline:
            break
        time.sleep(_ROLLOUT_SETTLE_INTERVAL_S)
    if latest is None:
        return result, None
    call_id, agents = latest
    evidence = _build_evidence(result, call_id=call_id, agents=agents)
    if evidence is None:
        return result, None
    return (
        replace(
            result,
            tool_calls=merge_tool_call_sequence(
                [*result.tool_calls, evidence]
            ),
        ),
        evidence,
    )


__all__ = ["enrich_codex_reconciliation_result"]
