"""AGY native ``swarm`` protocol parsing and reactive quiescence tracking."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import ACPEvent, ACPEventType

_SWARM_MANAGED_TOOL_NAMES = frozenset(
    {
        "swarm",
        "swarm_manage",
        "swarm_wait",
        "post_message",
        "swarm_post_message",
        "read_board",
        "swarm_read_board",
        "exit_plan_mode",
    }
)
_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])@(?:all|#[A-Za-z0-9_-]+|[A-Za-z0-9][A-Za-z0-9_-]*)"
)
_NARRATOR_FEEDBACK_RE = re.compile(
    r"^\s*@narrator(?:-\d+)?\s+user\s+feedback\b",
    re.IGNORECASE,
)
_AUXILIARY_PREFIX_RE = re.compile(r"^(?:monitor|narrator)-\d+$")
_TERMINAL_TOOL_STATUSES = frozenset({"completed", "failed"})


@dataclass(frozen=True)
class SwarmToolSignal:
    """Normalized swarm-orchestration signal parsed from an ACP tool update."""

    tool: str
    action: str = ""
    added_members: tuple[str, ...] = ()
    killed_targets: tuple[str, ...] = ()
    roster_members: tuple[tuple[str, str], ...] = ()
    is_done_post: bool = False
    mentions_worker: bool = False
    board_has_done: bool = False


def is_auxiliary_swarm_member(name: str) -> bool:
    """Return whether *name* identifies a swarm support role rather than a worker."""
    normalized = str(name or "").strip().casefold()
    if not normalized:
        return True
    if normalized in {"monitor", "narrator", "coordinator", "all"}:
        return True
    return bool(_AUXILIARY_PREFIX_RE.fullmatch(normalized))


def _unquote_arg(value: object) -> str:
    text = str(value or "").strip() if value is not None else ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                return decoded.strip()
        except Exception:
            return text[1:-1].strip()
    return text


def _normalize_tool_name(title: object) -> str:
    raw = str(title or "").strip().casefold()
    if not raw:
        return ""
    if raw.startswith("tool:"):
        raw = raw[5:].strip()
    if raw.startswith("run "):
        raw = raw[4:].strip()
    if " from mcp server " in raw:
        raw = raw.split(" from mcp server ", 1)[0].strip()
    if raw.startswith("contrib:"):
        raw = raw[len("contrib:") :].strip()
    if raw in {"swarm", "swarm_manage"}:
        return "swarm"
    if raw == "swarm_wait":
        return "swarm_wait"
    if raw in {"post_message", "swarm_post_message"}:
        return "post_message"
    if raw in {"read_board", "swarm_read_board"}:
        return "read_board"
    if raw == "exit_plan_mode":
        return "exit_plan_mode"
    return ""


def _extract_json_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    candidates: list[int] = []
    if text.startswith("{"):
        candidates.append(0)
    for idx, ch in enumerate(text):
        if ch == "{" and (idx == 0 or text[idx - 1] == "\n"):
            if idx not in candidates:
                candidates.append(idx)
            if len(candidates) >= 6:
                break
    if not candidates:
        first_brace = text.find("{")
        if first_brace >= 0:
            candidates.append(first_brace)
    for start in candidates:
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except Exception:
            continue
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return None


def _dedupe_names(values: Iterable[object]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = _unquote_arg(value)
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _extract_added_members(
    input_map: Mapping[str, Any] | None,
    output_map: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    names: list[str] = []
    if output_map is not None:
        for key in ("workers", "names"):
            raw_list = output_map.get(key)
            if isinstance(raw_list, (list, tuple)):
                names.extend(_dedupe_names(raw_list))
        out_name = _unquote_arg(output_map.get("name"))
        if out_name and not names:
            names.append(out_name)
    if not names and input_map is not None:
        base_name = _unquote_arg(input_map.get("name"))
        tasks = input_map.get("tasks")
        raw_n = input_map.get("n")
        if isinstance(tasks, (list, tuple)) and len(tasks) > 0 and base_name:
            names.extend(f"{base_name}-{idx}" for idx in range(1, len(tasks) + 1))
        elif (
            isinstance(raw_n, int)
            and not isinstance(raw_n, bool)
            and raw_n > 1
            and base_name
        ):
            names.extend(f"{base_name}-{idx}" for idx in range(1, raw_n + 1))
        elif base_name:
            names.append(base_name)
    return _dedupe_names(names)


def _extract_killed_targets(
    input_map: Mapping[str, Any] | None,
    output_map: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    targets: list[str] = []
    if input_map is not None:
        raw_name = _unquote_arg(input_map.get("name"))
        if raw_name:
            targets.append(raw_name)
    if output_map is not None:
        for key in ("killed", "names"):
            raw_list = output_map.get(key)
            if isinstance(raw_list, (list, tuple)):
                targets.extend(_dedupe_names(raw_list))
        raw_out_name = _unquote_arg(output_map.get("name"))
        if raw_out_name:
            targets.append(raw_out_name)
    return _dedupe_names(targets)


def _extract_roster_members(
    output_map: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if output_map is None:
        return ()
    raw_entries = output_map.get("members")
    if not isinstance(raw_entries, (list, tuple)):
        raw_entries = output_map.get("roster")
    if not isinstance(raw_entries, (list, tuple)):
        return ()
    roster: list[tuple[str, str]] = []
    for item in raw_entries:
        if not isinstance(item, Mapping):
            continue
        name = _unquote_arg(item.get("name"))
        state = _unquote_arg(item.get("state") or item.get("status")).casefold()
        if name and state:
            roster.append((name, state))
    return tuple(roster)


def _classify_post_message_text(text: str) -> tuple[bool, bool]:
    stripped = str(text or "").strip()
    if not stripped:
        return False, False
    mentions = _MENTION_RE.findall(stripped)
    is_done_post = stripped.casefold().startswith("[done]") and not mentions
    if is_done_post:
        return True, False
    if _NARRATOR_FEEDBACK_RE.match(stripped):
        return False, False
    return False, bool(mentions)


def parse_swarm_tool_signal(
    title: object,
    raw_input: object,
    raw_output: object,
) -> SwarmToolSignal | None:
    """Parse normalized AGY ``swarm`` tool signals from ACP metadata."""
    tool = _normalize_tool_name(title)
    input_map = _extract_json_mapping(raw_input)
    output_map = _extract_json_mapping(raw_output)

    if not tool:
        if output_map is not None:
            roster_members = _extract_roster_members(output_map)
            if roster_members:
                return SwarmToolSignal(
                    tool="",
                    action="status",
                    roster_members=roster_members,
                )
            out_status = _unquote_arg(output_map.get("status")).casefold()
            if out_status == "added":
                return SwarmToolSignal(
                    tool="",
                    action="add",
                    added_members=_extract_added_members(input_map, output_map),
                )
            if out_status == "killed":
                return SwarmToolSignal(
                    tool="",
                    action="kill",
                    killed_targets=_extract_killed_targets(input_map, output_map),
                )
        if isinstance(raw_output, str) and "[done]" in raw_output.casefold():
            return SwarmToolSignal(tool="", board_has_done=True)
        return None

    if tool == "swarm":
        action = _unquote_arg(
            (input_map or {}).get("action")
        ).casefold()
        if not action and output_map is not None:
            if _extract_roster_members(output_map):
                action = "status"
            else:
                out_status = _unquote_arg(output_map.get("status")).casefold()
                action = {
                    "added": "add",
                    "killed": "kill",
                    "titled": "title",
                }.get(out_status, "")
        added_members = (
            _extract_added_members(input_map, output_map)
            if action == "add"
            else ()
        )
        killed_targets = (
            _extract_killed_targets(input_map, output_map)
            if action == "kill"
            else ()
        )
        roster_members = (
            _extract_roster_members(output_map)
            if action in {"", "status"}
            else ()
        )
        return SwarmToolSignal(
            tool="swarm",
            action=action,
            added_members=added_members,
            killed_targets=killed_targets,
            roster_members=roster_members,
        )

    if tool == "post_message":
        raw_text = _unquote_arg((input_map or {}).get("text"))
        is_done_post, mentions_worker = _classify_post_message_text(raw_text)
        return SwarmToolSignal(
            tool="post_message",
            is_done_post=is_done_post,
            mentions_worker=mentions_worker,
        )

    if tool == "read_board":
        output_text = (
            raw_output
            if isinstance(raw_output, str)
            else json.dumps(raw_output, ensure_ascii=False)
            if raw_output is not None
            else ""
        )
        return SwarmToolSignal(
            tool="read_board",
            board_has_done="[done]" in output_text.casefold(),
        )

    if tool in {"swarm_wait", "exit_plan_mode"}:
        return SwarmToolSignal(tool=tool)

    return None


def merge_swarm_tool_signal(
    previous: SwarmToolSignal | None,
    current: SwarmToolSignal | None,
) -> SwarmToolSignal | None:
    """Combine start-frame inputs with progress/done-frame outputs."""
    if previous is None:
        return current
    if current is None:
        return previous
    return SwarmToolSignal(
        tool=current.tool or previous.tool,
        action=current.action or previous.action,
        added_members=_dedupe_names((*previous.added_members, *current.added_members)),
        killed_targets=_dedupe_names((*previous.killed_targets, *current.killed_targets)),
        roster_members=current.roster_members or previous.roster_members,
        is_done_post=previous.is_done_post or current.is_done_post,
        mentions_worker=previous.mentions_worker or current.mentions_worker,
        board_has_done=previous.board_has_done or current.board_has_done,
    )


def _signal_from_tool_call(tool_call: object) -> SwarmToolSignal | None:
    existing = getattr(tool_call, "swarm_signal", None)
    if isinstance(existing, SwarmToolSignal):
        return existing
    title = getattr(tool_call, "title", None)
    content = getattr(tool_call, "content", None)
    return parse_swarm_tool_signal(title, None, content)


def is_swarm_working_event(event: ACPEvent) -> bool:
    """Return True when an ACP event indicates active Swarm background execution."""
    tool_call = getattr(event, "tool_call", None)
    if tool_call is None:
        return False
    signal = _signal_from_tool_call(tool_call)
    if signal is None:
        return False
    status = str(getattr(tool_call, "status", "") or "").strip().casefold()
    event_type = getattr(event, "event_type", None)

    if (
        event_type is ACPEventType.TOOL_CALL_DONE
        and status == "completed"
        and signal.tool == "swarm"
        and signal.action == "add"
    ):
        return any(not is_auxiliary_swarm_member(member) for member in signal.added_members)

    if (
        event_type in {ACPEventType.TOOL_CALL_START, ACPEventType.TOOL_CALL_DONE}
        and status != "failed"
        and signal.tool == "swarm_wait"
    ):
        return True

    if (
        event_type is ACPEventType.TOOL_CALL_DONE
        and status == "completed"
        and signal.tool == "swarm"
        and signal.action == "status"
    ):
        return any(
            not is_auxiliary_swarm_member(name) and member_state == "working"
            for name, member_state in signal.roster_members
        )

    return False


class SwarmLifecycleTracker:
    """Track reactive AGY Swarm execution state across post-response wakeups."""

    def __init__(
        self,
        *,
        done_text_settle_s: float = 1.5,
        done_quiet_settle_s: float = 3.0,
        all_killed_text_settle_s: float = 2.0,
        idle_roster_text_settle_s: float = 3.0,
        turn_gap_reset_s: float = 2.0,
    ) -> None:
        self.done_text_settle_s = max(0.01, float(done_text_settle_s))
        self.done_quiet_settle_s = max(0.01, float(done_quiet_settle_s))
        self.all_killed_text_settle_s = max(0.01, float(all_killed_text_settle_s))
        self.idle_roster_text_settle_s = max(0.01, float(idle_roster_text_settle_s))
        self.turn_gap_reset_s = max(0.01, float(turn_gap_reset_s))

        self._signals_by_call_id: dict[str, SwarmToolSignal] = {}
        self._open_call_ids: set[str] = set()

        self.active = False
        self.done_posted = False
        self.active_task_workers: set[str] = set()
        self.ever_had_task_workers = False
        self.final_turn_text = ""

        self.prompt_response_returned = False
        self.awaiting_next_turn_wakeup = False
        self.burst_dispatched_work = False
        self.burst_parked_on_wait = False
        self.burst_plan_handoff = False
        self.burst_status_checked = False
        self.burst_working_task_workers = 0
        self.burst_saw_board_done = False
        self.has_text_since_last_tool = False
        self.last_activity_monotonic = 0.0
        self.last_text_monotonic = 0.0

    def _reset_burst_flags(self) -> None:
        self.awaiting_next_turn_wakeup = False
        self.burst_dispatched_work = False
        self.burst_parked_on_wait = False
        self.burst_plan_handoff = False
        self.burst_status_checked = False
        self.burst_working_task_workers = 0
        self.burst_saw_board_done = False
        self.has_text_since_last_tool = False
        self.final_turn_text = ""

    def _maybe_reset_burst(self, now_monotonic: float) -> None:
        if self.awaiting_next_turn_wakeup:
            self._reset_burst_flags()
            return
        if (
            self.prompt_response_returned
            and not self._open_call_ids
            and self.has_text_since_last_tool
            and (now_monotonic - self.last_activity_monotonic) >= self.turn_gap_reset_s
        ):
            self._reset_burst_flags()

    def _remove_killed_targets(self, targets: Iterable[str]) -> None:
        for raw_target in targets:
            target = str(raw_target or "").strip()
            if not target:
                continue
            prefix_re = re.compile(rf"^{re.escape(target)}-\d+$")
            to_remove = [
                member
                for member in self.active_task_workers
                if member == target or prefix_re.fullmatch(member)
            ]
            for member in to_remove:
                self.active_task_workers.discard(member)

    def _apply_roster(self, roster: tuple[tuple[str, str], ...]) -> None:
        self.burst_status_checked = True
        working_task_workers = 0
        for name, state in roster:
            if is_auxiliary_swarm_member(name):
                continue
            if state == "killed":
                self.active_task_workers.discard(name)
            elif state in {"working", "waiting", "stopped"}:
                self.active_task_workers.add(name)
                if state == "working":
                    working_task_workers += 1
        self.burst_working_task_workers = working_task_workers
        if working_task_workers > 0:
            self.active = True
            self.ever_had_task_workers = True

    def observe_event(self, event: ACPEvent, now_monotonic: float) -> bool:
        """Consume one ACPEvent and return True when Swarm lifecycle state changed."""
        prev_active = self.active
        prev_done = self.done_posted
        prev_waiting = self.awaiting_next_turn_wakeup
        prev_workers = frozenset(self.active_task_workers)

        if self.last_activity_monotonic == 0.0:
            self.last_activity_monotonic = now_monotonic
        self._maybe_reset_burst(now_monotonic)
        self.last_activity_monotonic = now_monotonic

        event_type = getattr(event, "event_type", None)
        if event_type is ACPEventType.TEXT_CHUNK:
            chunk = getattr(event, "text", None) or ""
            if chunk:
                self.has_text_since_last_tool = True
                self.final_turn_text += chunk
                self.last_text_monotonic = now_monotonic
        elif event_type in {
            ACPEventType.TOOL_CALL_START,
            ACPEventType.TOOL_CALL_UPDATE,
            ACPEventType.TOOL_CALL_DONE,
        }:
            tool_call = getattr(event, "tool_call", None)
            if tool_call is not None:
                self.has_text_since_last_tool = False
                self.final_turn_text = ""

                call_id = str(getattr(tool_call, "id", "") or "").strip()
                status = str(getattr(tool_call, "status", "") or "").strip().casefold()
                incoming_sig = _signal_from_tool_call(tool_call)
                prev_sig = self._signals_by_call_id.get(call_id) if call_id else None
                merged_sig = merge_swarm_tool_signal(prev_sig, incoming_sig)
                if call_id and merged_sig is not None:
                    self._signals_by_call_id[call_id] = merged_sig
                if merged_sig is not None and hasattr(tool_call, "swarm_signal"):
                    tool_call.swarm_signal = merged_sig

                is_terminal = (
                    event_type is ACPEventType.TOOL_CALL_DONE
                    or status in _TERMINAL_TOOL_STATUSES
                )
                if call_id:
                    if is_terminal:
                        self._open_call_ids.discard(call_id)
                    else:
                        self._open_call_ids.add(call_id)

                if is_terminal and status == "completed" and merged_sig is not None:
                    if merged_sig.tool == "swarm":
                        if merged_sig.action == "add":
                            task_workers = [
                                name
                                for name in merged_sig.added_members
                                if not is_auxiliary_swarm_member(name)
                            ]
                            if task_workers:
                                self.active_task_workers.update(task_workers)
                                self.ever_had_task_workers = True
                                self.active = True
                                self.done_posted = False
                                self.burst_dispatched_work = True
                        elif merged_sig.action == "kill":
                            self._remove_killed_targets(merged_sig.killed_targets)
                            if (
                                self.ever_had_task_workers
                                and not self.active_task_workers
                            ):
                                self.burst_dispatched_work = False
                        elif merged_sig.action == "model":
                            if self.active:
                                self.burst_dispatched_work = True
                        elif merged_sig.action == "status":
                            self._apply_roster(merged_sig.roster_members)
                    elif merged_sig.tool == "swarm_wait":
                        self.active = True
                        self.burst_parked_on_wait = True
                        self.awaiting_next_turn_wakeup = True
                    elif merged_sig.tool == "post_message":
                        if merged_sig.is_done_post:
                            self.done_posted = True
                        elif merged_sig.mentions_worker:
                            self.burst_dispatched_work = True
                    elif merged_sig.tool == "read_board":
                        if merged_sig.board_has_done:
                            self.burst_saw_board_done = True
                    elif merged_sig.tool == "exit_plan_mode":
                        if self.active:
                            self.burst_plan_handoff = True
                            self.awaiting_next_turn_wakeup = True

        return (
            self.active != prev_active
            or self.done_posted != prev_done
            or self.awaiting_next_turn_wakeup != prev_waiting
            or frozenset(self.active_task_workers) != prev_workers
        )

    def on_prompt_response_returned(self, now_monotonic: float) -> None:
        """Record that the initial ACP ``session/prompt`` RPC returned."""
        self.prompt_response_returned = True
        if self.last_activity_monotonic == 0.0:
            self.last_activity_monotonic = now_monotonic
        if (
            self.active
            and not self.done_posted
            and (
                self.burst_dispatched_work
                or self.burst_parked_on_wait
                or self.burst_plan_handoff
            )
        ):
            self.awaiting_next_turn_wakeup = True

    def evaluate_wait(
        self,
        now_monotonic: float,
        *,
        has_incomplete_outer_tools: bool = False,
    ) -> tuple[bool, float | None]:
        """Return ``(should_wait, next_check_timeout_s)`` for quiescence polling."""
        if not self.active:
            return False, None
        if has_incomplete_outer_tools or bool(self._open_call_ids):
            return True, None

        if self.done_posted and not self.burst_parked_on_wait and not self.burst_plan_handoff:
            target_settle = (
                self.done_text_settle_s
                if self.has_text_since_last_tool
                else self.done_quiet_settle_s
            )
            elapsed = max(0.0, now_monotonic - self.last_activity_monotonic)
            if elapsed >= target_settle:
                return False, None
            return True, target_settle - elapsed

        if (
            self.awaiting_next_turn_wakeup
            or self.burst_parked_on_wait
            or self.burst_plan_handoff
            or self.burst_dispatched_work
        ):
            return True, None

        if (
            self.ever_had_task_workers
            and not self.active_task_workers
            and self.has_text_since_last_tool
        ):
            elapsed = max(0.0, now_monotonic - self.last_activity_monotonic)
            if elapsed >= self.all_killed_text_settle_s:
                return False, None
            return True, self.all_killed_text_settle_s - elapsed

        if (
            self.prompt_response_returned
            and self.has_text_since_last_tool
            and (
                (self.burst_status_checked and self.burst_working_task_workers == 0)
                or (self.burst_saw_board_done and len(self.active_task_workers) <= 1)
            )
        ):
            elapsed = max(0.0, now_monotonic - self.last_activity_monotonic)
            if elapsed >= self.idle_roster_text_settle_s:
                return False, None
            return True, self.idle_roster_text_settle_s - elapsed

        return True, None


__all__ = [
    "SwarmLifecycleTracker",
    "SwarmToolSignal",
    "is_auxiliary_swarm_member",
    "is_swarm_working_event",
    "merge_swarm_tool_signal",
    "parse_swarm_tool_signal",
]
