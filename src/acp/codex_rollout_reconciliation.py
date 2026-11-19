"""Strict Codex rollout fallback for omitted terminal observations.

The official Codex ACP adapter can omit raw generic function completions and
authoritative collaboration snapshots from its live stream. This module reads
only the active session's bounded rollout tail, then projects exact matching
function output and fully attributable child lifecycle evidence into canonical
ACP tool calls. Any identity, time-window, or shape ambiguity fails closed.
"""

from __future__ import annotations

import json
import os
import re
import stat
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .collaboration import merge_tool_call_sequence
from .models import PromptResult, ToolCallInfo
from .outcome import has_transient_child_lifecycle

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9-]{16,128}")
_MAX_HEAD_BYTES = 1024 * 1024
_MAX_TAIL_BYTES = 4 * 1024 * 1024
_MAX_FUNCTION_OUTPUT_BYTES = 1024 * 1024
_ROLLOUT_SETTLE_TIMEOUT_S = 0.2
_ROLLOUT_SETTLE_INTERVAL_S = 0.02
CODEX_CHILD_LIFECYCLE_POLL_INTERVAL_S = 0.1
_CHILD_TURN_BOUNDARY_SLACK_S = 1.0
_MAX_TRACKED_CHILDREN = 200
_MAX_AGENT_DEPTH = 8
_MAX_INDEXED_ROOTS = 16
_MAX_INDEXED_DIRS_PER_ROOT = 2048
_MAX_INDEXED_FILES_PER_ROOT = 20_000
_HOT_DIR_SETTLE_NS = 50_000_000
_ROOT_AGENT_PATH = "/root"
_AGENT_PATH_RE = re.compile(
    r"/root(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}){1,8}"
)
_SUBAGENT_ACTIVITY_KINDS = frozenset(
    {"started", "interacted", "interrupted"}
)
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TRANSIENT_OUTER_STATUSES = frozenset(
    {"pending", "in_progress", "running"}
)
_KNOWN_COLLAB_SPAWN_FAILURES = frozenset(
    {
        "Full-history forked agents inherit the parent agent type; omit "
        "agent_type, or spawn without a full-history fork.",
        "collab spawn failed: agent thread limit reached",
    }
)
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


@dataclass(frozen=True)
class _DirRolloutIndex:
    sig: tuple[int, int]
    scanned_at_ns: int
    by_tail: dict[str, tuple[str, ...]]
    file_count: int


@dataclass
class _RootRolloutIndex:
    dirs: OrderedDict[tuple[str, str, str], _DirRolloutIndex] = field(
        default_factory=OrderedDict
    )
    total_files: int = 0


_ROLLOUT_INDEX: OrderedDict[Path, _RootRolloutIndex] = OrderedDict()
_ROLLOUT_INDEX_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _uuidv7_partition_rel_dirs(
    session_id: str,
) -> tuple[tuple[str, str, str], ...] | None:
    """Return candidate YYYY/MM/DD partitions across UTC and local +-1d."""
    try:
        parsed = uuid.UUID(session_id)
    except ValueError:
        return None
    if parsed.version != 7:
        return None
    timestamp_s = (parsed.int >> 80) / 1000.0
    try:
        utc_dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        local_dt = utc_dt.astimezone()
    except (OverflowError, OSError, ValueError):
        return None
    seen: dict[tuple[str, str, str], None] = {}
    for base_dt in (utc_dt, local_dt):
        base_date = base_dt.date()
        for day_offset in (-1, 0, 1):
            try:
                target_date = base_date + timedelta(days=day_offset)
            except OverflowError:
                continue
            seen.setdefault(
                (
                    f"{target_date.year:04d}",
                    f"{target_date.month:02d}",
                    f"{target_date.day:02d}",
                ),
                None,
            )
    return tuple(seen) or None


def _iter_digit_subdirs(parent: Path, width: int) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    try:
        with os.scandir(parent) as it:
            for entry in it:
                name = entry.name
                if len(name) != width or not name.isdigit():
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                items.append((name, Path(entry.path)))
    except OSError:
        return []
    return items


def _list_partition_keys(
    resolved_root: Path,
) -> tuple[tuple[str, str, str], ...]:
    keys: list[tuple[str, str, str]] = []
    for year, year_path in _iter_digit_subdirs(resolved_root, 4):
        for month, month_path in _iter_digit_subdirs(year_path, 2):
            for day, _day_path in _iter_digit_subdirs(month_path, 2):
                keys.append((year, month, day))
    return tuple(keys)


def _scan_partition_dir(day_dir: Path) -> _DirRolloutIndex | None:
    try:
        st = os.stat(day_dir, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISDIR(st.st_mode):
        return None
    buckets: dict[str, list[str]] = {}
    file_count = 0
    try:
        with os.scandir(day_dir) as it:
            for entry in it:
                name = entry.name
                # Minimum valid "rollout--<16-char-id>.jsonl" is 31 chars.
                if (
                    len(name) < 31
                    or not name.startswith("rollout-")
                    or not name.endswith(".jsonl")
                ):
                    continue
                try:
                    if entry.is_symlink() or not entry.is_file(
                        follow_symlinks=False
                    ):
                        continue
                except OSError:
                    continue
                buckets.setdefault(name[-22:-6], []).append(name)
                file_count += 1
    except OSError:
        return None
    return _DirRolloutIndex(
        sig=(st.st_mtime_ns, st.st_ino),
        scanned_at_ns=time.time_ns(),
        by_tail={tail: tuple(names) for tail, names in buckets.items()},
        file_count=file_count,
    )


def _evict_partition_dir(
    resolved_root: Path, day_key: tuple[str, str, str]
) -> None:
    with _ROLLOUT_INDEX_LOCK:
        root_idx = _ROLLOUT_INDEX.get(resolved_root)
        if root_idx is None:
            return
        removed = root_idx.dirs.pop(day_key, None)
        if removed is not None:
            root_idx.total_files = max(
                0, root_idx.total_files - removed.file_count
            )


def _store_partition_dir(
    resolved_root: Path,
    day_key: tuple[str, str, str],
    scanned: _DirRolloutIndex,
) -> None:
    with _ROLLOUT_INDEX_LOCK:
        root_idx = _ROLLOUT_INDEX.get(resolved_root)
        if root_idx is None:
            while len(_ROLLOUT_INDEX) >= _MAX_INDEXED_ROOTS:
                _ROLLOUT_INDEX.popitem(last=False)
            root_idx = _RootRolloutIndex()
            _ROLLOUT_INDEX[resolved_root] = root_idx
        else:
            _ROLLOUT_INDEX.move_to_end(resolved_root)
        old = root_idx.dirs.pop(day_key, None)
        if old is not None:
            root_idx.total_files = max(0, root_idx.total_files - old.file_count)
        if scanned.file_count > _MAX_INDEXED_FILES_PER_ROOT:
            return
        while root_idx.dirs and (
            len(root_idx.dirs) >= _MAX_INDEXED_DIRS_PER_ROOT
            or root_idx.total_files + scanned.file_count
            > _MAX_INDEXED_FILES_PER_ROOT
        ):
            _, evicted = root_idx.dirs.popitem(last=False)
            root_idx.total_files = max(
                0, root_idx.total_files - evicted.file_count
            )
        root_idx.dirs[day_key] = scanned
        root_idx.total_files += scanned.file_count


def _lookup_in_partition_dir(
    resolved_root: Path,
    day_key: tuple[str, str, str],
    session_id: str,
) -> list[Path]:
    year_dir = resolved_root / day_key[0]
    month_dir = year_dir / day_key[1]
    day_dir = month_dir / day_key[2]
    try:
        if (
            year_dir.is_symlink()
            or month_dir.is_symlink()
            or day_dir.is_symlink()
        ):
            _evict_partition_dir(resolved_root, day_key)
            return []
        st = os.stat(day_dir, follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode):
            _evict_partition_dir(resolved_root, day_key)
            return []
        resolved_day = day_dir.resolve(strict=True)
        if not resolved_day.is_relative_to(resolved_root):
            _evict_partition_dir(resolved_root, day_key)
            return []
    except OSError:
        _evict_partition_dir(resolved_root, day_key)
        return []

    dir_sig = (st.st_mtime_ns, st.st_ino)
    tail16 = session_id[-16:]
    suffix = f"-{session_id}.jsonl"
    min_len = 8 + len(suffix)

    for attempt in range(2):
        cached_names: tuple[str, ...] | None = None
        if attempt == 0:
            with _ROLLOUT_INDEX_LOCK:
                root_idx = _ROLLOUT_INDEX.get(resolved_root)
                if root_idx is not None:
                    _ROLLOUT_INDEX.move_to_end(resolved_root)
                    entry = root_idx.dirs.get(day_key)
                    if entry is not None and entry.sig == dir_sig:
                        root_idx.dirs.move_to_end(day_key)
                        matched = entry.by_tail.get(tail16)
                        if (
                            matched is not None
                            or abs(entry.scanned_at_ns - st.st_mtime_ns)
                            >= _HOT_DIR_SETTLE_NS
                        ):
                            cached_names = matched or ()
        if cached_names is None:
            scanned = _scan_partition_dir(day_dir)
            if scanned is None:
                _evict_partition_dir(resolved_root, day_key)
                return []
            _store_partition_dir(resolved_root, day_key, scanned)
            cached_names = scanned.by_tail.get(tail16, ())

        found: list[Path] = []
        stale = False
        for name in cached_names:
            if len(name) < min_len or not name.endswith(suffix):
                continue
            candidate = day_dir / name
            try:
                resolved = candidate.resolve(strict=True)
                if candidate.is_symlink() or not resolved.is_relative_to(
                    resolved_root
                ):
                    continue
            except OSError:
                stale = True
                continue
            found.append(resolved)
        if not stale:
            return found
        _evict_partition_dir(resolved_root, day_key)
    return []


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
    except OSError:
        return set()

    partition_keys = _uuidv7_partition_rel_dirs(session_id)
    if partition_keys is None:
        partition_keys = _list_partition_keys(resolved_root)

    matches: set[Path] = set()
    for day_key in partition_keys:
        matches.update(
            _lookup_in_partition_dir(resolved_root, day_key, session_id)
        )
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


def _pending_outer_tool_calls(
    result: PromptResult,
) -> dict[str, ToolCallInfo]:
    """Return known transient outer calls keyed by their provider identity."""
    pending: dict[str, ToolCallInfo] = {}
    for tool_call in result.tool_calls:
        if not isinstance(tool_call, ToolCallInfo):
            continue
        raw_call_id = tool_call.id
        raw_status = tool_call.status
        raw_kind = tool_call.kind
        if not all(
            isinstance(value, str)
            for value in (raw_call_id, raw_status, raw_kind)
        ):
            continue
        if not raw_call_id or raw_call_id != raw_call_id.strip():
            continue
        call_id = raw_call_id
        status = raw_status.strip().casefold()
        kind = raw_kind.strip().casefold()
        if (
            call_id
            and kind == "other"
            and status in _TRANSIENT_OUTER_STATUSES
            and tool_call.is_context_compaction is not True
        ):
            pending[call_id] = tool_call
    return pending


def _is_bounded_generic_function_output(payload: Mapping[str, Any]) -> bool:
    """Accept only the generic output shapes the Codex ACP adapter supports."""
    if "output" not in payload:
        return False
    output = payload.get("output")
    if isinstance(output, str):
        return len(output.encode("utf-8")) <= _MAX_FUNCTION_OUTPUT_BYTES
    if not isinstance(output, list):
        return False
    try:
        encoded = json.dumps(output, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return False
    return len(encoded) <= _MAX_FUNCTION_OUTPUT_BYTES


def _collaboration_spawn_terminal_status(output: object) -> str | None:
    """Classify only Codex's strictly known spawn success/failure outputs."""
    if not isinstance(output, str):
        return None
    if len(output.encode("utf-8")) > _MAX_FUNCTION_OUTPUT_BYTES:
        return None
    text = output.strip()
    if text in _KNOWN_COLLAB_SPAWN_FAILURES:
        return "failed"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or set(value) != {"task_name"}:
        return None
    task_name = value.get("task_name")
    if not isinstance(task_name, str):
        return None
    return "completed" if _AGENT_PATH_RE.fullmatch(task_name.strip()) else None


def _generic_function_terminal_status(
    *,
    name: str,
    namespace: str | None,
    output: object,
) -> str | None:
    """Map a paired generic function output without guessing tool failures."""
    if name == "spawn_agent":
        if namespace != "collaboration":
            return None
        return _collaboration_spawn_terminal_status(output)
    return "completed"


def _function_call_output_evidence(
    path: Path,
    result: PromptResult,
    *,
    started_at: float,
    ended_at: float,
) -> tuple[ToolCallInfo, ...]:
    """Recover only current-turn generic calls with an exact terminal output.

    Codex's live ACP item stream can omit completion updates for generic
    ``function_call`` records (notably the built-in ``wait`` tool).  A matching
    ``function_call_output`` in the unique parent rollout is an authoritative
    terminal observation, but it must be paired with the same call id in this
    prompt's bounded time window.  Anything malformed, incomplete, or outside
    that window remains unresolved.
    """
    if ended_at < started_at:
        return ()
    pending = _pending_outer_tool_calls(result)
    if not pending:
        return ()
    lines, complete = _read_bounded_lines(path, tail=True)
    if not complete:
        return ()

    calls: dict[str, list[tuple[float, int, str, str | None]]] = {}
    outputs: dict[str, list[tuple[float, int, object]]] = {}
    invalid_calls: set[str] = set()
    for line_index, raw_line in enumerate(lines):
        event = _decode_line(raw_line)
        if event is None:
            return ()
        if event.get("type") != "response_item":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        item_type = payload.get("type")
        if item_type not in {"function_call", "function_call_output"}:
            continue
        raw_call_id = payload.get("call_id")
        if (
            not isinstance(raw_call_id, str)
            or not raw_call_id
            or raw_call_id != raw_call_id.strip()
        ):
            continue
        call_id = raw_call_id
        if call_id not in pending:
            continue
        timestamp = _event_timestamp(event.get("timestamp"))
        if timestamp is None:
            invalid_calls.add(call_id)
            continue
        if timestamp < started_at or timestamp > ended_at:
            continue
        if item_type == "function_call":
            raw_name = payload.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                invalid_calls.add(call_id)
                continue
            raw_namespace = payload.get("namespace")
            namespace = (
                raw_namespace.strip()
                if isinstance(raw_namespace, str) and raw_namespace.strip()
                else None
            )
            calls.setdefault(call_id, []).append(
                (timestamp, line_index, raw_name.strip(), namespace)
            )
            continue
        if not _is_bounded_generic_function_output(payload):
            invalid_calls.add(call_id)
            continue
        outputs.setdefault(call_id, []).append(
            (timestamp, line_index, payload.get("output"))
        )

    evidence: list[ToolCallInfo] = []
    for call_id, tool_call in pending.items():
        candidates = calls.get(call_id, [])
        terminal_outputs = outputs.get(call_id, [])
        if (
            call_id in invalid_calls
            or len(candidates) != 1
            or len(terminal_outputs) != 1
        ):
            continue
        call_timestamp, call_index, name, namespace = candidates[0]
        output_timestamp, output_index, output = terminal_outputs[0]
        if output_timestamp < call_timestamp or output_index <= call_index:
            continue
        status = _generic_function_terminal_status(
            name=name,
            namespace=namespace,
            output=output,
        )
        if status is not None:
            evidence.append(replace(tool_call, status=status))
    return tuple(evidence)


def _append_evidence(
    result: PromptResult,
    evidence: tuple[ToolCallInfo, ...],
) -> PromptResult:
    if not evidence:
        return result
    return replace(
        result,
        tool_calls=merge_tool_call_sequence([*result.tool_calls, *evidence]),
    )


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
    owner_agent_path: str = _ROOT_AGENT_PATH,
    target_sources: frozenset[str] | None = None,
    started_at: float,
    ended_at: float,
) -> tuple[dict[str, str], bool, bool]:
    """Read unique descendant-child identities from one bounded parent tail."""
    del target_sources  # Closure wave validation checks full coverage across files.
    if ended_at < started_at:
        return {}, True, False
    owner_prefix = f"{owner_agent_path}/"
    source_to_path: dict[str, str] = {}
    path_to_source: dict[str, str] = {}
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
            or not isinstance(raw_kind, str)
            or raw_kind.strip().casefold() not in _SUBAGENT_ACTIVITY_KINDS
        ):
            return {}, True, False
        source = raw_source.strip()
        agent_path = raw_agent_path.strip()
        if (
            agent_path != _ROOT_AGENT_PATH
            and _AGENT_PATH_RE.fullmatch(agent_path) is None
        ):
            return {}, True, False
        # Ignore sibling/ancestor/self activities recorded in this rollout.
        if not agent_path.startswith(owner_prefix):
            continue
        if (
            source in source_to_path
            and source_to_path[source] != agent_path
        ) or (
            agent_path in path_to_source
            and path_to_source[agent_path] != source
        ):
            return {}, True, False
        if (
            source not in source_to_path
            and len(source_to_path) >= _MAX_TRACKED_CHILDREN
        ):
            return {}, True, False
        source_to_path[source] = agent_path
        path_to_source[agent_path] = source
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


def _nested_child_meta_path(
    path: Path,
    *,
    parent_session_id: str,
    child_thread_id: str,
    cwd: str,
    known_thread_paths: Mapping[str, str],
    invalid_sources: set[str],
) -> str | None:
    """Resolve an explicit nested child path from verified session_meta."""
    payload = _session_meta_payload(path)
    if payload is None:
        return None
    raw_cwd = payload.get("cwd")
    if (
        payload.get("id") != child_thread_id
        or payload.get("session_id") != parent_session_id
        or not isinstance(raw_cwd, str)
        or os.path.realpath(raw_cwd) != os.path.realpath(cwd)
        or not _is_subagent_session_meta(payload)
    ):
        return None
    raw_path = payload.get("agent_path")
    if not isinstance(raw_path, str):
        return None
    recorded_path = raw_path.strip()
    if _AGENT_PATH_RE.fullmatch(recorded_path) is None:
        return None
    expected_parent_path = recorded_path.rpartition("/")[0]
    # Depth-1 direct children must be anchored by ACP or root rollout activity.
    if expected_parent_path == _ROOT_AGENT_PATH:
        return None
    parent_thread_id = payload.get("parent_thread_id")
    if (
        not isinstance(parent_thread_id, str)
        or parent_thread_id == parent_session_id
        or parent_thread_id in invalid_sources
        or known_thread_paths.get(parent_thread_id) != expected_parent_path
    ):
        return None
    return recorded_path


def _file_stat_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


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
        self._observed_sources: set[str] = set()
        self._source_to_path: dict[str, str] = {}
        self._path_to_source: dict[str, str] = {}
        self._rollout_by_source: dict[str, Path] = {}
        self._invalid_sources: set[str] = set()
        self._next_lookup_at: dict[str, float] = {}
        self._last_fingerprint: tuple[tuple[str, str, str], ...] | None = None
        self._saw_rollout_candidate = False
        self._identity_ambiguous = False
        self._activity_ambiguous = False
        self._retryable_activity_read = False
        self._root_rollout: Path | None = None
        self._root_rollout_next_lookup_at = 0.0
        self._root_activity_sources: dict[str, str] = {}
        self._activity_scan_stats: dict[str, tuple[int, int]] = {}

    @property
    def saw_rollout_candidate(self) -> bool:
        return self._saw_rollout_candidate

    @property
    def activity_ambiguous(self) -> bool:
        return self._activity_ambiguous

    @property
    def retryable_activity_read(self) -> bool:
        return self._retryable_activity_read

    @property
    def resolved_path_to_source(self) -> dict[str, str]:
        return dict(self._path_to_source)

    def set_root_rollout(self, rollout: Path | None) -> None:
        self._root_rollout = rollout
        self._root_rollout_next_lookup_at = float("inf")

    def _register_identity(
        self,
        source: str,
        agent_path: str,
        *,
        from_activity: bool = True,
    ) -> bool:
        """Register a validated 1-to-1 source_id <-> agent_path mapping."""
        if (
            _SESSION_ID_RE.fullmatch(source) is None
            or _AGENT_PATH_RE.fullmatch(agent_path) is None
            or source == self._parent_session_id
        ):
            if from_activity:
                self._activity_ambiguous = True
            self._identity_ambiguous = True
            return False
        existing = self._source_to_path.get(source)
        if existing is not None and existing != agent_path:
            if from_activity:
                self._activity_ambiguous = True
            self._identity_ambiguous = True
            return False
        existing_source = self._path_to_source.get(agent_path)
        if existing_source is not None and existing_source != source:
            if from_activity:
                self._activity_ambiguous = True
            self._identity_ambiguous = True
            return False
        if (
            source not in self._source_to_path
            and len(self._source_to_path) >= _MAX_TRACKED_CHILDREN
        ):
            if from_activity:
                self._activity_ambiguous = True
            self._identity_ambiguous = True
            return False
        if source in self._source_to_path:
            return False
        self._source_to_path[source] = agent_path
        self._path_to_source[agent_path] = source
        return True

    def _record_observed_source(self, raw_source: object) -> str | None:
        if (
            not isinstance(raw_source, str)
            or _SESSION_ID_RE.fullmatch(raw_source.strip()) is None
        ):
            self._identity_ambiguous = True
            return None
        source = raw_source.strip()
        if source == self._parent_session_id:
            self._identity_ambiguous = True
            return None
        if (
            source not in self._observed_sources
            and len(self._observed_sources) >= _MAX_TRACKED_CHILDREN
        ):
            self._identity_ambiguous = True
            return None
        self._observed_sources.add(source)
        return source

    def observe_tool_call(self, tool_call: object) -> None:
        raw_source = getattr(tool_call, "subagent_source_id", None)
        raw_path = getattr(tool_call, "subagent_path", None)
        if raw_source not in (None, "") or raw_path not in (None, ""):
            source = self._record_observed_source(raw_source)
            if source is None:
                return
            if raw_path not in (None, ""):
                if not isinstance(raw_path, str):
                    self._identity_ambiguous = True
                    return
                self._register_identity(
                    source,
                    raw_path.strip(),
                    from_activity=False,
                )
                if self._identity_ambiguous:
                    return

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
                self._identity_ambiguous = True
                return
            for item in raw_container:
                val = item.get("source_id") if isinstance(item, Mapping) else item
                if self._record_observed_source(val) is None:
                    return

    def observe_result(self, result: PromptResult) -> None:
        for tool_call in result.tool_calls:
            self.observe_tool_call(tool_call)

    def _scan_activity_file(
        self,
        *,
        owner_id: str,
        owner_path: str,
        rollout_path: Path,
        observed_until: float,
    ) -> dict[str, str] | None:
        stat_sig = _file_stat_sig(rollout_path)
        if (
            stat_sig is not None
            and stat_sig == self._activity_scan_stats.get(owner_id)
        ):
            return None
        activities, ambiguous, retryable = _parent_activity_paths(
            rollout_path,
            owner_agent_path=owner_path,
            started_at=self._logical_task_started_at,
            ended_at=observed_until,
        )
        if retryable:
            self._retryable_activity_read = True
            return None
        if ambiguous:
            self._activity_ambiguous = True
            self._identity_ambiguous = True
            return None
        if stat_sig is not None:
            self._activity_scan_stats[owner_id] = stat_sig
        return activities

    def _validate_known_rollout(
        self,
        source_id: str,
        agent_path: str,
        *,
        known_thread_paths: Mapping[str, str],
        now: float,
        force_lookup: bool,
    ) -> bool:
        if (
            source_id in self._rollout_by_source
            or source_id in self._invalid_sources
            or (
                not force_lookup
                and now < self._next_lookup_at.get(source_id, 0.0)
            )
        ):
            return False
        expected_parent_path = agent_path.rpartition("/")[0]
        if (
            expected_parent_path != _ROOT_AGENT_PATH
            and expected_parent_path not in self._path_to_source
        ):
            return False
        candidates = _rollout_candidates(
            session_id=source_id,
            codex_home=self._codex_home,
        )
        if not candidates:
            self._next_lookup_at[source_id] = now + 0.25
            return False
        self._saw_rollout_candidate = True
        if len(candidates) != 1:
            self._invalid_sources.add(source_id)
            return False
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
            return False
        self._rollout_by_source[source_id] = candidate
        return True

    def _has_unaccounted_root_activity(self) -> bool:
        if not self._root_activity_sources:
            return False
        target_paths = [
            self._source_to_path[src]
            for src in self._observed_sources
            if src in self._source_to_path
        ]
        for src, path in self._root_activity_sources.items():
            if src in self._observed_sources:
                continue
            prefix = f"{path}/"
            if any(target_path.startswith(prefix) for target_path in target_paths):
                continue
            return True
        return False

    def _locate_rollouts(
        self,
        *,
        observed_until: float | None = None,
        force_lookup: bool = False,
    ) -> None:
        until = time.time() if observed_until is None else observed_until
        now = time.monotonic()
        self._retryable_activity_read = False

        if (
            self._root_rollout is None
            and (self._observed_sources - set(self._source_to_path))
            and (force_lookup or now >= self._root_rollout_next_lookup_at)
        ):
            found_root = _find_rollout(
                session_id=self._parent_session_id,
                cwd=self._cwd,
                codex_home=self._codex_home,
            )
            if found_root is not None:
                self._root_rollout = found_root
            else:
                self._root_rollout_next_lookup_at = now + 0.25

        if self._root_rollout is not None:
            root_activities = self._scan_activity_file(
                owner_id=self._parent_session_id,
                owner_path=_ROOT_AGENT_PATH,
                rollout_path=self._root_rollout,
                observed_until=until,
            )
            if self._retryable_activity_read or self._activity_ambiguous:
                return
            if root_activities is not None:
                self._root_activity_sources = dict(root_activities)
                for src, agent_path in root_activities.items():
                    self._register_identity(src, agent_path)
                    if self._activity_ambiguous:
                        return

        for _ in range(_MAX_AGENT_DEPTH + 1):
            progressed = False
            known_thread_paths = {
                self._parent_session_id: _ROOT_AGENT_PATH,
                **self._source_to_path,
            }
            for source_id, agent_path in list(self._source_to_path.items()):
                if self._validate_known_rollout(
                    source_id,
                    agent_path,
                    known_thread_paths=known_thread_paths,
                    now=now,
                    force_lookup=force_lookup,
                ):
                    progressed = True

            for owner_id, owner_rollout in list(self._rollout_by_source.items()):
                owner_path = self._source_to_path.get(owner_id)
                if (
                    owner_path is None
                    or owner_path.count("/") - 1 >= _MAX_AGENT_DEPTH
                ):
                    continue
                child_activities = self._scan_activity_file(
                    owner_id=owner_id,
                    owner_path=owner_path,
                    rollout_path=owner_rollout,
                    observed_until=until,
                )
                if self._retryable_activity_read or self._activity_ambiguous:
                    return
                if child_activities:
                    for child_id, child_path in child_activities.items():
                        if self._register_identity(child_id, child_path):
                            progressed = True
                        if self._activity_ambiguous:
                            return

            unresolved = sorted(
                self._observed_sources - set(self._source_to_path)
            )
            if unresolved:
                known_thread_paths = {
                    self._parent_session_id: _ROOT_AGENT_PATH,
                    **self._source_to_path,
                }
                for source_id in unresolved:
                    if (
                        source_id in self._invalid_sources
                        or (
                            not force_lookup
                            and now < self._next_lookup_at.get(source_id, 0.0)
                        )
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
                    recorded_path = _nested_child_meta_path(
                        candidate,
                        parent_session_id=self._parent_session_id,
                        child_thread_id=source_id,
                        cwd=self._cwd,
                        known_thread_paths=known_thread_paths,
                        invalid_sources=self._invalid_sources,
                    )
                    if recorded_path is None:
                        continue
                    if self._register_identity(source_id, recorded_path):
                        progressed = True
                    if self._activity_ambiguous:
                        return
                    self._rollout_by_source[source_id] = candidate
                    progressed = True

            if self._invalid_sources or not progressed:
                break

    def poll(
        self,
        *,
        ended_at: float | None = None,
        require_all_terminal: bool = False,
        emit_unchanged: bool = False,
    ) -> ToolCallInfo | None:
        if self._identity_ambiguous or (
            not self._source_to_path and not self._observed_sources
        ):
            return None
        observed_until = time.time() if ended_at is None else ended_at
        self._locate_rollouts(
            observed_until=observed_until,
            force_lookup=require_all_terminal,
        )
        if (
            self._invalid_sources
            or self._activity_ambiguous
            or self._retryable_activity_read
        ):
            return None
        if not self._source_to_path:
            return None
        if require_all_terminal and (
            self._has_unaccounted_root_activity()
            or not self._observed_sources.issubset(self._source_to_path)
        ):
            return None
        tracked_sources = (
            self._observed_sources
            if self._observed_sources
            else set(self._source_to_path)
        )
        observations: list[tuple[str, _ChildTurnObservation]] = []
        for source_id in self._source_to_path:
            if source_id not in tracked_sources:
                continue
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
            len(observations) != len(tracked_sources)
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


def _merge_path_to_source(
    base: Mapping[str, str] | None,
    extra: Mapping[str, str],
) -> dict[str, str] | None:
    path_to_source: dict[str, str] = {}
    source_to_path: dict[str, str] = {}
    for source_map in (base or {}, extra):
        for agent_path, source_id in source_map.items():
            existing_src = path_to_source.get(agent_path)
            if existing_src is not None and existing_src != source_id:
                return None
            existing_path = source_to_path.get(source_id)
            if existing_path is not None and existing_path != agent_path:
                return None
            path_to_source[agent_path] = source_id
            source_to_path[source_id] = agent_path
    return path_to_source or None


def _build_evidence(
    result: PromptResult,
    *,
    call_id: str,
    agents: list[Mapping[str, Any]],
    path_to_source: Mapping[str, str] | None = None,
) -> ToolCallInfo | None:
    resolved_path_to_source = (
        dict(path_to_source)
        if path_to_source is not None
        else _path_to_source(result)
    )
    if not resolved_path_to_source:
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
        if name in resolved_path_to_source:
            if status not in _TERMINAL_STATUSES:
                return None
            observed[name] = status
            continue
        return None
    if root_count != 1:
        return None
    if set(observed) != set(resolved_path_to_source):
        return None
    ordered_paths = tuple(resolved_path_to_source)
    return ToolCallInfo(
        id=f"rollout-list-agents:{call_id}",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        collaboration_receivers=tuple(
            resolved_path_to_source[path] for path in ordered_paths
        ),
        subagent_states=tuple(
            {
                "source_id": resolved_path_to_source[path],
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
    target_sources, malformed_sources = _result_child_sources(result)
    if malformed_sources:
        return result, None
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
    monitor.set_root_rollout(rollout)
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
        if monitor.activity_ambiguous:
            return result, None
        if (
            monitor._identity_ambiguous
            and not monitor.retryable_activity_read
        ):
            break
        if time.monotonic() >= settle_deadline:
            break
        time.sleep(_ROLLOUT_SETTLE_INTERVAL_S)
    if (
        monitor.saw_rollout_candidate
        or monitor.activity_ambiguous
        or monitor.retryable_activity_read
        or monitor._has_unaccounted_root_activity()
    ):
        return result, None
    if rollout is None:
        return result, None
    merged_path_to_source = _merge_path_to_source(
        _path_to_source(result),
        monitor.resolved_path_to_source,
    )
    if merged_path_to_source is None:
        return result, None
    if target_sources and not target_sources.issubset(
        set(merged_path_to_source.values())
    ):
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
    evidence = _build_evidence(
        result,
        call_id=call_id,
        agents=agents,
        path_to_source=merged_path_to_source,
    )
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


def enrich_codex_terminal_result(
    result: PromptResult,
    *,
    session_id: str,
    cwd: str,
    logical_task_started_at: float | None = None,
    started_at: float,
    ended_at: float,
    codex_home: str | None = None,
) -> tuple[PromptResult, tuple[ToolCallInfo, ...]]:
    """Recover omitted Codex terminal evidence before outcome classification.

    This is deliberately an observation-only boundary hook.  It first folds
    exact generic function output pairs from the uniquely attributable parent
    rollout, then reuses the strict child-turn reconciliation path.  No missing
    record, ambiguous identity, or partial rollout can turn a transient call
    into a terminal one.
    """
    pending_outer = _pending_outer_tool_calls(result)
    has_transient_children = has_transient_child_lifecycle(result.tool_calls)
    if not pending_outer and not has_transient_children:
        return result, ()
    evidence: list[ToolCallInfo] = []
    enriched = result
    if pending_outer:
        rollout = _find_rollout(
            session_id=session_id,
            cwd=cwd,
            codex_home=codex_home,
        )
    else:
        rollout = None
    if rollout is not None:
        settle_deadline = time.monotonic() + _ROLLOUT_SETTLE_TIMEOUT_S
        while True:
            generic_evidence = _function_call_output_evidence(
                rollout,
                enriched,
                started_at=started_at,
                ended_at=ended_at,
            )
            if generic_evidence:
                enriched = _append_evidence(enriched, generic_evidence)
                evidence.extend(generic_evidence)
            if not _pending_outer_tool_calls(enriched):
                break
            if time.monotonic() >= settle_deadline:
                break
            time.sleep(_ROLLOUT_SETTLE_INTERVAL_S)

    child_sources, child_sources_malformed = _result_child_sources(enriched)
    if child_sources and not child_sources_malformed:
        try:
            child_enriched, child_evidence = enrich_codex_reconciliation_result(
                enriched,
                session_id=session_id,
                cwd=cwd,
                logical_task_started_at=logical_task_started_at,
                started_at=started_at,
                ended_at=ended_at,
                codex_home=codex_home,
            )
        except (OSError, RuntimeError, ValueError):
            return enriched, tuple(evidence)
        if child_evidence is not None:
            enriched = child_enriched
            evidence.append(child_evidence)
    return enriched, tuple(evidence)


__all__ = [
    "CodexChildLifecycleMonitor",
    "enrich_codex_reconciliation_result",
    "enrich_codex_terminal_result",
]
