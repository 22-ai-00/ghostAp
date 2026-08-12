"""Strict helpers for folding ACP child-agent lifecycle snapshots."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from .models import ToolCallInfo

TRANSIENT_CHILD_STATUSES = frozenset({"pending", "running"})
TERMINAL_CHILD_STATUSES = frozenset(
    {"completed", "failed", "cancelled"}
)
KNOWN_CHILD_STATUSES = TRANSIENT_CHILD_STATUSES | TERMINAL_CHILD_STATUSES
_CHILD_GENERATION_ACTIONS = frozenset({"followup_task", "spawn_agent"})
_PASSIVE_CHILD_ACTIONS = frozenset(
    {"interrupt_agent", "list_agents", "send_message", "wait_agent"}
)
_KNOWN_CHILD_ACTIONS = _CHILD_GENERATION_ACTIONS | _PASSIVE_CHILD_ACTIONS
_KNOWN_CHILD_ACTIVITIES = frozenset(
    {"interacted", "interrupted", "started"}
)


class AuthoritativeChildLifecycleProof:
    """Track a final explicit ``list_agents`` proof outside presentation code.

    The proof is invalidated whenever a known child starts a new generation or
    emits later activity. Malformed identities/statuses remain fail-closed.
    """

    def __init__(self) -> None:
        self._observed_sources: set[str] = set()
        self._authoritative_statuses: dict[str, str] = {}
        self._ambiguous = False

    def observe_event(self, event: object) -> None:
        tool_call = getattr(event, "tool_call", None)
        if tool_call is None:
            return
        event_name = str(
            getattr(getattr(event, "event_type", None), "name", "") or ""
        )
        outer_status = str(
            getattr(tool_call, "status", "") or ""
        ).strip().casefold()
        action = _child_action(tool_call)
        activity = str(
            getattr(tool_call, "subagent_activity", "") or ""
        ).strip().casefold()
        receiver_items, receivers_malformed = _receiver_items(
            getattr(tool_call, "collaboration_receivers", ())
        )
        receivers: set[str] = set()
        for raw_receiver in receiver_items:
            source_id = strict_source_id(raw_receiver)
            if source_id is None:
                receivers_malformed = True
            else:
                receivers.add(source_id)

        states: dict[str, str] = {}
        states_malformed = False
        for item in _state_items(getattr(tool_call, "subagent_states", ())):
            if not isinstance(item, Mapping):
                states_malformed = True
                continue
            source_id = strict_source_id(item.get("source_id"))
            status = strict_child_status(item.get("status"))
            if source_id is None or status is None:
                states_malformed = True
                continue
            prior = states.get(source_id)
            if prior is not None and prior != status:
                states_malformed = True
                continue
            states[source_id] = status

        activity_source = strict_source_id(
            getattr(tool_call, "subagent_source_id", None)
        )
        if activity and (
            activity not in _KNOWN_CHILD_ACTIVITIES
            or activity_source is None
        ):
            states_malformed = True

        sources = {*receivers, *states}
        if activity_source is not None:
            sources.add(activity_source)
        self._observed_sources.update(sources)

        has_child_evidence = bool(
            sources
            or activity
            or getattr(tool_call, "child_metadata_malformed", False)
        )
        known_activity_action = action.startswith("activity:")
        if (
            receivers_malformed
            or states_malformed
            or getattr(tool_call, "child_metadata_malformed", False) is not False
            or (
                has_child_evidence
                and action not in _KNOWN_CHILD_ACTIONS
                and not known_activity_action
            )
        ):
            self._ambiguous = True
            self._authoritative_statuses.clear()

        starts_generation = (
            event_name == "TOOL_CALL_DONE"
            and outer_status == "completed"
            and action in _CHILD_GENERATION_ACTIONS
        )
        if starts_generation or activity in _KNOWN_CHILD_ACTIVITIES:
            for source_id in sources:
                self._authoritative_statuses.pop(source_id, None)

        if action != "list_agents" or event_name != "TOOL_CALL_DONE":
            return
        if outer_status != "completed":
            return
        if (
            receivers_malformed
            or states_malformed
            or not states
            or not receivers.issubset(states)
        ):
            self._ambiguous = True
            self._authoritative_statuses.clear()
            return
        self._authoritative_statuses = states

    def all_observed_children_terminal(self) -> bool:
        """Whether the latest clean list snapshot covers all known children."""
        if self._ambiguous or not self._observed_sources:
            return False
        if not self._observed_sources.issubset(self._authoritative_statuses):
            return False
        return all(
            self._authoritative_statuses[source_id]
            in TERMINAL_CHILD_STATUSES
            for source_id in self._observed_sources
        )


def strict_source_id(value: object) -> str | None:
    """Return a validated provider child identity without coercing values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def strict_child_status(value: object) -> str | None:
    """Return a known lifecycle status; unknown provider values stay invalid."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in KNOWN_CHILD_STATUSES else None


def _state_items(raw_states: object) -> list[object]:
    if raw_states is None:
        return []
    if isinstance(raw_states, (str, bytes, bytearray, Mapping)):
        return [
            {"source_id": "", "status": "malformed", "message": ""}
        ]
    if not isinstance(raw_states, Iterable):
        return [
            {"source_id": "", "status": "malformed", "message": ""}
        ]
    return list(raw_states)


def _receiver_items(raw_receivers: object) -> tuple[list[object], bool]:
    if isinstance(raw_receivers, (str, bytes, bytearray, Mapping)):
        return [], True
    if not isinstance(raw_receivers, Iterable):
        return [], True
    return list(raw_receivers), False


def _snapshot_has_malformed_child_metadata(tool_call: ToolCallInfo) -> bool:
    marker = getattr(tool_call, "child_metadata_malformed", False)
    if marker is not False:
        return True
    collaboration_tool = tool_call.collaboration_tool
    if collaboration_tool is not None and not isinstance(
        collaboration_tool,
        str,
    ):
        return True
    receiver_items, malformed_receivers = _receiver_items(
        tool_call.collaboration_receivers
    )
    if malformed_receivers or any(
        strict_source_id(receiver) is None for receiver in receiver_items
    ):
        return True
    raw_source_id = tool_call.subagent_source_id
    raw_activity = tool_call.subagent_activity
    if raw_source_id not in (None, "") or raw_activity not in (None, ""):
        if strict_source_id(raw_source_id) is None:
            return True
        if not isinstance(raw_activity, str):
            return True
        if raw_activity.strip().casefold() not in _KNOWN_CHILD_ACTIVITIES:
            return True
    return False


def _child_action(tool_call: ToolCallInfo) -> str:
    collaboration_tool = tool_call.collaboration_tool
    if isinstance(collaboration_tool, str) and collaboration_tool.strip():
        return collaboration_tool.strip().casefold()
    title = tool_call.title
    if isinstance(title, str):
        normalized_title = title.strip().casefold()
        if normalized_title in _KNOWN_CHILD_ACTIONS:
            return normalized_title
    activity = tool_call.subagent_activity
    if isinstance(activity, str):
        normalized_activity = activity.strip().casefold()
        if normalized_activity in _KNOWN_CHILD_ACTIVITIES:
            return f"activity:{normalized_activity}"
    return ""


def _generation_action(tool_call: ToolCallInfo) -> str:
    action = _child_action(tool_call)
    return action if action in _CHILD_GENERATION_ACTIONS else ""


def _activity_is_transient(tool_call: ToolCallInfo) -> bool:
    activity = tool_call.subagent_activity
    normalized = (
        activity.strip().casefold() if isinstance(activity, str) else ""
    )
    return normalized in {"started", "interacted"} or (
        normalized == "interrupted"
        and str(tool_call.status or "").strip().casefold() != "completed"
    )


def _unknown_action_has_child_evidence(tool_call: ToolCallInfo) -> bool:
    collaboration_tool = tool_call.collaboration_tool
    normalized = (
        collaboration_tool.strip().casefold()
        if isinstance(collaboration_tool, str)
        else ""
    )
    if not normalized or normalized in _KNOWN_CHILD_ACTIONS:
        return False
    receiver_items, _ = _receiver_items(tool_call.collaboration_receivers)
    return bool(
        receiver_items
        or _state_items(tool_call.subagent_states)
        or tool_call.subagent_source_id
        or tool_call.subagent_activity
    )


def _reopens_child_generation(tool_call: ToolCallInfo) -> bool:
    return bool(
        _generation_action(tool_call)
        or _activity_is_transient(tool_call)
        or _unknown_action_has_child_evidence(tool_call)
    )


def _has_child_lifecycle_observation(tool_call: ToolCallInfo) -> bool:
    """Return whether a snapshot carries child state, receiver, or activity."""
    receiver_items, malformed_receivers = _receiver_items(
        tool_call.collaboration_receivers
    )
    return bool(
        _state_items(tool_call.subagent_states)
        or receiver_items
        or malformed_receivers
        or tool_call.subagent_source_id not in (None, "")
        or tool_call.subagent_activity not in (None, "")
        or getattr(tool_call, "child_metadata_malformed", False) is not False
        or _snapshot_has_malformed_child_metadata(tool_call)
    )


def _is_child_related(tool_call: ToolCallInfo) -> bool:
    return bool(
        _child_action(tool_call)
        or _has_child_lifecycle_observation(tool_call)
    )


def merge_child_state_snapshots(
    previous: object,
    current: object,
    *,
    reset_sources: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    """Merge updates for one outer call without erasing invalid evidence.

    A terminal observation is sticky only against recognized stale transient
    observations. Unknown identities, statuses, and malformed entries remain as
    independent fail-closed evidence for the outcome classifier.
    """
    known_by_source: dict[str, Mapping[str, object]] = {}
    invalid: list[object] = []

    def add_invalid(item: object) -> None:
        if item not in invalid:
            invalid.append(item)

    def ingest(item: object) -> None:
        if not isinstance(item, Mapping):
            add_invalid(item)
            return
        source_id = strict_source_id(item.get("source_id"))
        status = strict_child_status(item.get("status"))
        if source_id is None or status is None:
            add_invalid(item)
            return
        prior = known_by_source.get(source_id)
        prior_status = (
            strict_child_status(prior.get("status"))
            if prior is not None
            else None
        )
        if (
            prior_status in TERMINAL_CHILD_STATUSES
            and status in TRANSIENT_CHILD_STATUSES
        ):
            return
        known_by_source[source_id] = item

    for state in _state_items(previous):
        ingest(state)
    for source_id in reset_sources:
        known_by_source.pop(source_id, None)
    for state in _state_items(current):
        ingest(state)
    return tuple([*known_by_source.values(), *invalid])


def merge_tool_call_snapshot(
    previous: ToolCallInfo,
    current: ToolCallInfo,
) -> ToolCallInfo:
    """Merge two snapshots for the same outer tool-call identity."""
    receivers: list[str] = []
    previous_receiver_items, previous_receivers_malformed = _receiver_items(
        previous.collaboration_receivers
    )
    current_receiver_items, current_receivers_malformed = _receiver_items(
        current.collaboration_receivers
    )
    for raw_receiver in (
        *previous_receiver_items,
        *current_receiver_items,
    ):
        receiver = strict_source_id(raw_receiver)
        if receiver is not None and receiver not in receivers:
            receivers.append(receiver)
    current_tool = _generation_action(current)
    previous_tool = _generation_action(previous)
    reset_sources: set[str] = set()
    if (
        (
            current_tool in _CHILD_GENERATION_ACTIONS
            and previous_tool != current_tool
        )
        or _activity_is_transient(current)
        or _unknown_action_has_child_evidence(current)
    ):
        reset_sources.update(
            source_id
            for raw_receiver in current_receiver_items
            if (source_id := strict_source_id(raw_receiver)) is not None
        )
        for state in _state_items(current.subagent_states):
            if not isinstance(state, Mapping):
                continue
            source_id = strict_source_id(state.get("source_id"))
            if source_id is not None:
                reset_sources.add(source_id)
        activity_source = strict_source_id(current.subagent_source_id)
        if activity_source is not None:
            reset_sources.add(activity_source)
    return replace(
        current,
        collaboration_tool=(
            current.collaboration_tool or previous.collaboration_tool
        ),
        collaboration_receivers=tuple(receivers),
        collaboration_model=(
            current.collaboration_model or previous.collaboration_model
        ),
        subagent_source_id=(
            current.subagent_source_id or previous.subagent_source_id
        ),
        subagent_path=current.subagent_path or previous.subagent_path,
        subagent_activity=(
            current.subagent_activity or previous.subagent_activity
        ),
        subagent_states=merge_child_state_snapshots(
            previous.subagent_states,
            current.subagent_states,
            reset_sources=frozenset(reset_sources),
        ),
        child_metadata_malformed=(
            previous_receivers_malformed
            or current_receivers_malformed
            or _snapshot_has_malformed_child_metadata(previous)
            or _snapshot_has_malformed_child_metadata(current)
        ),
    )


def merge_tool_call_sequence(
    tool_calls: Iterable[ToolCallInfo],
    *,
    generation_boundary_index: int | None = None,
) -> list[Any]:
    """Fold named snapshots while retaining semantic invocation chronology.

    Updates for the same named invocation keep their first position, so a stale
    duplicate cannot jump past newer lifecycle evidence.  An action change,
    generation-introducing enrichment, or cross-prompt reuse creates a distinct
    invocation slot even when the provider reuses the same id. Anonymous calls
    always receive distinct slots.
    """
    ordered: OrderedDict[tuple[str, object], Any] = OrderedDict()
    anonymous_index = 0
    named_generation = 0
    active_key_by_id: dict[str, tuple[str, object]] = {}
    boundary_seen_ids: set[str] = set()
    child_epoch = 0
    child_epoch_by_id: dict[str, int] = {}
    for index, tool_call in enumerate(tool_calls):
        if isinstance(tool_call, ToolCallInfo) and tool_call.id:
            key = active_key_by_id.get(
                tool_call.id,
                ("named", tool_call.id),
            )
            previous = ordered.get(key)
            crosses_turn_boundary = False
            if (
                generation_boundary_index is not None
                and index >= generation_boundary_index
                and tool_call.id not in boundary_seen_ids
            ):
                crosses_turn_boundary = previous is not None
                boundary_seen_ids.add(tool_call.id)
            previous_action = (
                _child_action(previous) if previous is not None else ""
            )
            current_action = _child_action(tool_call)
            action_changed = (
                previous is not None
                and bool(previous_action)
                and bool(current_action)
                and current_action != previous_action
            )
            introduces_reopening_action = (
                previous is not None
                and not previous_action
                and _reopens_child_generation(tool_call)
            )
            current_has_lifecycle = _has_child_lifecycle_observation(
                tool_call
            )
            intervening_child_call = (
                previous is not None
                and child_epoch
                > child_epoch_by_id.get(tool_call.id, child_epoch)
            )
            boundary_starts_child_invocation = (
                crosses_turn_boundary
                and previous is not None
                and (
                    _is_child_related(previous)
                    or _is_child_related(tool_call)
                )
                and (
                    bool(_generation_action(tool_call))
                    or not current_has_lifecycle
                    or intervening_child_call
                )
            )
            starts_new_invocation = (
                boundary_starts_child_invocation
                or action_changed
                or introduces_reopening_action
            )
            if starts_new_invocation:
                key = (
                    "named-generation",
                    (tool_call.id, named_generation),
                )
                named_generation += 1
                active_key_by_id[tool_call.id] = key
                previous = None
            else:
                active_key_by_id[tool_call.id] = key
            merged = (
                tool_call
                if previous is None
                or (
                    not _is_child_related(previous)
                    and not _is_child_related(tool_call)
                )
                else merge_tool_call_snapshot(previous, tool_call)
            )
            ordered[key] = merged
            if _is_child_related(tool_call):
                child_epoch += 1
            child_epoch_by_id[tool_call.id] = child_epoch
            continue
        key = ("anonymous", anonymous_index)
        anonymous_index += 1
        ordered[key] = tool_call
    return list(ordered.values())


__all__ = [
    "AuthoritativeChildLifecycleProof",
    "KNOWN_CHILD_STATUSES",
    "TERMINAL_CHILD_STATUSES",
    "TRANSIENT_CHILD_STATUSES",
    "merge_child_state_snapshots",
    "merge_tool_call_sequence",
    "merge_tool_call_snapshot",
    "strict_child_status",
    "strict_source_id",
]
