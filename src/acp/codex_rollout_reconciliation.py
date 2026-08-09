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
from collections.abc import Mapping
from dataclasses import replace
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
_ROOT_AGENT_PATH = "/root"
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


def _read_bounded_lines(path: Path, *, tail: bool) -> list[bytes]:
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
                        return []
            else:
                data = handle.read(limit)
    except OSError:
        return []
    return data.splitlines()


def _decode_line(raw_line: bytes) -> Mapping[str, Any] | None:
    try:
        value = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _session_matches(path: Path, session_id: str, cwd: str) -> bool:
    expected_cwd = os.path.realpath(cwd)
    for raw_line in _read_bounded_lines(path, tail=False):
        event = _decode_line(raw_line)
        if event is None or event.get("type") != "session_meta":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            return False
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
    return False


def _find_rollout(
    *,
    session_id: str,
    cwd: str,
    codex_home: str | None,
) -> Path | None:
    if _SESSION_ID_RE.fullmatch(session_id) is None:
        return None
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
        return None
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
        if _session_matches(resolved, session_id, cwd):
            matches.add(resolved)
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
    calls: list[tuple[float, int, object]] = []
    outputs: list[tuple[float, int, str, object]] = []
    for line_index, raw_line in enumerate(
        _read_bounded_lines(path, tail=True)
    ):
        event = _decode_line(raw_line)
        if event is None or event.get("type") != "response_item":
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
                    (timestamp, line_index, payload.get("call_id"))
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
    call_timestamp, call_index, raw_call_id = max(
        calls,
        key=lambda item: (item[0], item[1]),
    )
    if not isinstance(raw_call_id, str) or not raw_call_id.strip():
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
            continue
        if name in path_to_source:
            if status not in _TERMINAL_STATUSES:
                return None
            observed[name] = status
            continue
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
    started_at: float,
    ended_at: float,
    codex_home: str | None = None,
) -> tuple[PromptResult, ToolCallInfo | None]:
    """Add one attributable rollout snapshot, otherwise return unchanged."""
    rollout = _find_rollout(
        session_id=session_id,
        cwd=cwd,
        codex_home=codex_home,
    )
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
