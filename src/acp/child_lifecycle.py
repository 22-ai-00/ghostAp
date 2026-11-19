"""Shared pure child-agent lifecycle reducer and status normalization.

Used by both ACP completion classification (``outcome``, ``collaboration``,
``codex_rollout_reconciliation``) and card event projection (``projector``)
so status vocabulary, generation reopening, sticky malformed tracking, and
outer-vs-child authority boundaries have one source of truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum

TRANSIENT_CHILD_STATUSES = frozenset({"pending", "running"})
TERMINAL_CHILD_STATUSES = frozenset({"completed", "failed", "cancelled"})
KNOWN_CHILD_STATUSES = TRANSIENT_CHILD_STATUSES | TERMINAL_CHILD_STATUSES

CHILD_GENERATION_ACTIONS = frozenset({"followup_task", "spawn_agent"})
PASSIVE_CHILD_ACTIONS = frozenset(
    {"interrupt_agent", "list_agents", "send_message", "wait_agent"}
)
KNOWN_CHILD_ACTIONS = CHILD_GENERATION_ACTIONS | PASSIVE_CHILD_ACTIONS
KNOWN_CHILD_ACTIVITIES = frozenset({"interacted", "interrupted", "started"})

_PROVIDER_STATUS_MAP = {
    "pending": "pending",
    "pendinginit": "pending",
    "pending_init": "pending",
    "running": "running",
    "in_progress": "running",
    "completed": "completed",
    "failed": "failed",
    "errored": "failed",
    "notfound": "failed",
    "not_found": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "interrupted": "cancelled",
    "shutdown": "cancelled",
}


class ChildObservationProvenance(str, Enum):
    """Authority source of a child lifecycle observation."""

    COLLABORATION_SNAPSHOT = "collaboration_snapshot"
    AUTHORITATIVE_LIST = "authoritative_list"
    SUBAGENT_ACTIVITY = "subagent_activity"
    ROLLOUT_TURN = "rollout_turn"


def strict_source_id(value: object) -> str | None:
    """Return a validated provider child identity without coercing values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def normalize_provider_child_status(value: object) -> str | None:
    """Map a provider child status string into canonical lifecycle vocabulary."""
    if not isinstance(value, str):
        return None
    return _PROVIDER_STATUS_MAP.get(value.strip().casefold())


def iter_state_items(raw_states: object) -> tuple[list[object], bool]:
    """Return raw child state entries and whether the container is malformed."""
    if raw_states is None:
        return [], False
    if isinstance(raw_states, (str, bytes, bytearray, Mapping)) or not isinstance(
        raw_states, Iterable
    ):
        return [], True
    return list(raw_states), False


def iter_receiver_items(raw_receivers: object) -> tuple[list[object], bool]:
    """Return raw receiver thread items and whether the container is malformed."""
    if raw_receivers is None:
        return [], False
    if isinstance(
        raw_receivers, (str, bytes, bytearray, Mapping)
    ) or not isinstance(raw_receivers, Iterable):
        return [], True
    return list(raw_receivers), False


def collaboration_action_name(tool_call: object) -> str:
    """Return normalized collaboration action name from tool metadata or title."""
    collaboration_tool = getattr(tool_call, "collaboration_tool", None)
    if isinstance(collaboration_tool, str) and collaboration_tool.strip():
        return collaboration_tool.strip().casefold()
    title = getattr(tool_call, "title", None)
    if isinstance(title, str):
        normalized_title = title.strip().casefold()
        if normalized_title in KNOWN_CHILD_ACTIONS:
            return normalized_title
    return ""


def child_action_key(tool_call: object) -> str:
    """Return collaboration action or ``activity:<kind>`` identity key."""
    action = collaboration_action_name(tool_call)
    if action:
        return action
    activity = getattr(tool_call, "subagent_activity", None)
    if isinstance(activity, str):
        normalized_activity = activity.strip().casefold()
        if normalized_activity in KNOWN_CHILD_ACTIVITIES:
            return f"activity:{normalized_activity}"
    return ""


@dataclass(frozen=True)
class ParsedChildObservation:
    """One child observation extracted from an ACP tool or rollout event."""

    source_id: str
    status: str | None
    raw_status: str
    provenance: ChildObservationProvenance
    message: str = ""
    malformed: bool = False


@dataclass(frozen=True)
class ParsedChildToolCall:
    """Normalized child lifecycle payload extracted from one tool snapshot."""

    call_id: str
    action: str
    outer_status: str
    event_name: str
    activity: str
    activity_source: str | None
    receivers: tuple[str, ...]
    observations: tuple[ParsedChildObservation, ...]
    malformed: bool
    starts_new_generation: bool
    authoritative_list_snapshot: bool
    reopens_transient: bool
    activity_is_transient: bool
    failed_generation_with_activity: bool
    has_child_evidence: bool
    container_malformed: bool = False


@dataclass(frozen=True)
class ChildLifecycleEntry:
    """Reduced lifecycle state for one stable child thread identity."""

    source_id: str
    status: str | None
    raw_status: str
    generation: int
    provenance: ChildObservationProvenance
    malformed: bool = False
    last_index: int = -1
    message: str = ""
    active_call_id: str = ""

    @property
    def is_terminal(self) -> bool:
        return not self.malformed and self.status in TERMINAL_CHILD_STATUSES

    @property
    def is_transient(self) -> bool:
        return not self.malformed and self.status in TRANSIENT_CHILD_STATUSES


def parse_child_tool_call(
    tool_call: object,
    *,
    event_name: str = "",
) -> ParsedChildToolCall:
    """Parse one tool call into normalized child observations and flags."""
    call_id = str(getattr(tool_call, "id", "") or "")
    outer_status = str(getattr(tool_call, "status", "") or "").strip().casefold()
    normalized_event = str(event_name or "").strip().upper()
    is_done_frame = not normalized_event or normalized_event == "TOOL_CALL_DONE"

    container_malformed = (
        getattr(tool_call, "child_metadata_malformed", False) is not False
    )
    raw_collab_tool = getattr(tool_call, "collaboration_tool", None)
    if raw_collab_tool is not None and not isinstance(raw_collab_tool, str):
        container_malformed = True
    malformed = container_malformed
    action = collaboration_action_name(tool_call)

    authoritative_list = (
        action == "list_agents" and outer_status == "completed" and is_done_frame
    )
    if authoritative_list and call_id.startswith("rollout-child-lifecycle:"):
        snapshot_provenance = ChildObservationProvenance.ROLLOUT_TURN
    elif authoritative_list:
        snapshot_provenance = ChildObservationProvenance.AUTHORITATIVE_LIST
    else:
        snapshot_provenance = ChildObservationProvenance.COLLABORATION_SNAPSHOT

    state_items, states_container_malformed = iter_state_items(
        getattr(tool_call, "subagent_states", None)
    )
    if states_container_malformed:
        container_malformed = True
        malformed = True

    observations: list[ParsedChildObservation] = []
    represented_sources: set[str] = set()
    seen_statuses: dict[str, str | None] = {}
    for item in state_items:
        if not isinstance(item, Mapping):
            malformed = True
            continue
        source_id = strict_source_id(item.get("source_id"))
        raw_status_obj = item.get("status")
        raw_status_str = (
            raw_status_obj.strip().casefold()
            if isinstance(raw_status_obj, str)
            else ""
        )
        norm_status = normalize_provider_child_status(raw_status_obj)
        raw_msg = item.get("message")
        message = raw_msg if isinstance(raw_msg, str) else str(raw_msg or "")
        if source_id is None or norm_status is None:
            malformed = True
            if source_id is not None:
                represented_sources.add(source_id)
                observations.append(
                    ParsedChildObservation(
                        source_id=source_id,
                        status=None,
                        raw_status=raw_status_str,
                        provenance=snapshot_provenance,
                        message=message,
                        malformed=True,
                    )
                )
            continue
        prior_in_call = seen_statuses.get(source_id)
        if prior_in_call is not None and prior_in_call != norm_status:
            malformed = True
        seen_statuses[source_id] = norm_status
        represented_sources.add(source_id)
        observations.append(
            ParsedChildObservation(
                source_id=source_id,
                status=norm_status,
                raw_status=raw_status_str,
                provenance=snapshot_provenance,
                message=message,
                malformed=False,
            )
        )

    receiver_items, receivers_container_malformed = iter_receiver_items(
        getattr(tool_call, "collaboration_receivers", ())
    )
    if receivers_container_malformed:
        container_malformed = True
        malformed = True
    receivers: list[str] = []
    for raw_receiver in receiver_items:
        receiver = strict_source_id(raw_receiver)
        if receiver is None:
            malformed = True
            continue
        if receiver not in receivers:
            receivers.append(receiver)
        if receiver not in represented_sources:
            observations.append(
                ParsedChildObservation(
                    source_id=receiver,
                    status="pending",
                    raw_status="pending",
                    provenance=snapshot_provenance,
                )
            )

    raw_activity = getattr(tool_call, "subagent_activity", None)
    raw_activity_source = getattr(tool_call, "subagent_source_id", None)
    activity = ""
    activity_source: str | None = None
    if raw_activity not in (None, "") or raw_activity_source not in (None, ""):
        activity_source = strict_source_id(raw_activity_source)
        activity = (
            raw_activity.strip().casefold()
            if isinstance(raw_activity, str)
            else ""
        )
        if activity_source is None or activity not in KNOWN_CHILD_ACTIVITIES:
            malformed = True
        elif activity_source not in represented_sources:
            if activity in {"started", "interacted"}:
                act_status = "running"
                act_raw = "running"
            else:
                act_status = (
                    "cancelled" if outer_status == "completed" else "running"
                )
                act_raw = (
                    "interrupted" if outer_status == "completed" else "running"
                )
            observations.append(
                ParsedChildObservation(
                    source_id=activity_source,
                    status=act_status,
                    raw_status=act_raw,
                    provenance=ChildObservationProvenance.SUBAGENT_ACTIVITY,
                )
            )

    starts_new_generation = (
        action in CHILD_GENERATION_ACTIONS
        and outer_status == "completed"
        and is_done_frame
    )
    if starts_new_generation and not observations:
        malformed = True

    activity_is_transient = activity in {"started", "interacted"} or (
        activity == "interrupted" and outer_status != "completed"
    )
    has_child_evidence = bool(
        observations
        or receivers
        or activity
        or activity_source is not None
        or malformed
    )
    reopens_transient = bool(
        action and action not in PASSIVE_CHILD_ACTIONS and has_child_evidence
    )
    failed_generation_with_activity = (
        action in CHILD_GENERATION_ACTIONS
        and outer_status != "completed"
        and activity in KNOWN_CHILD_ACTIVITIES
        and activity_source is not None
    )

    return ParsedChildToolCall(
        call_id=call_id,
        action=action,
        outer_status=outer_status,
        event_name=normalized_event,
        activity=activity,
        activity_source=activity_source,
        receivers=tuple(receivers),
        observations=tuple(observations),
        malformed=malformed,
        starts_new_generation=starts_new_generation,
        authoritative_list_snapshot=authoritative_list,
        reopens_transient=reopens_transient,
        activity_is_transient=activity_is_transient,
        failed_generation_with_activity=failed_generation_with_activity,
        has_child_evidence=has_child_evidence,
        container_malformed=container_malformed,
    )


def reduce_child_entry(
    prior: ChildLifecycleEntry | None,
    *,
    source_id: str,
    status: str | None,
    raw_status: str = "",
    provenance: ChildObservationProvenance,
    malformed: bool = False,
    reopen: bool = False,
    reopen_while_transient: bool = False,
    call_id: str = "",
    index: int = -1,
    message: str = "",
) -> ChildLifecycleEntry:
    """Apply one child observation onto prior state via shared lifecycle rules."""
    if prior is None:
        if malformed or status is None:
            return ChildLifecycleEntry(
                source_id=source_id,
                status=None,
                raw_status=raw_status,
                generation=1,
                provenance=provenance,
                malformed=True,
                last_index=index,
                message=message,
                active_call_id=call_id,
            )
        return ChildLifecycleEntry(
            source_id=source_id,
            status=status,
            raw_status=raw_status or status,
            generation=1,
            provenance=provenance,
            malformed=False,
            last_index=index,
            message=message,
            active_call_id=call_id,
        )

    # Sticky malformed: keep any prior transient index so fail-closed
    # diagnostics preserve both earlier running and later malformed calls.
    if malformed or status is None or prior.malformed:
        return replace(
            prior,
            malformed=True,
            raw_status=raw_status or prior.raw_status,
            provenance=provenance,
            message=message or prior.message,
        )

    should_reopen = reopen and (
        prior.status in TERMINAL_CHILD_STATUSES
        or (
            reopen_while_transient
            and bool(call_id)
            and bool(prior.active_call_id)
            and call_id != prior.active_call_id
        )
    )
    if should_reopen:
        return ChildLifecycleEntry(
            source_id=source_id,
            status=status,
            raw_status=raw_status or status,
            generation=prior.generation + 1,
            provenance=provenance,
            malformed=False,
            last_index=index,
            message=message,
            active_call_id=call_id or prior.active_call_id,
        )

    if (
        prior.status in TERMINAL_CHILD_STATUSES
        and status in TRANSIENT_CHILD_STATUSES
    ):
        return prior

    # Do not let a passive snapshot downgrade an already running child to pending.
    if (
        prior.status == "running"
        and status == "pending"
        and provenance is ChildObservationProvenance.COLLABORATION_SNAPSHOT
    ):
        return ChildLifecycleEntry(
            source_id=source_id,
            status=prior.status,
            raw_status=prior.raw_status,
            generation=prior.generation,
            provenance=prior.provenance,
            malformed=False,
            last_index=prior.last_index if prior.last_index >= 0 else index,
            message=message or prior.message,
            active_call_id=prior.active_call_id or call_id,
        )

    return ChildLifecycleEntry(
        source_id=source_id,
        status=status,
        raw_status=raw_status or status,
        generation=prior.generation,
        provenance=provenance,
        malformed=False,
        last_index=index if index >= 0 else prior.last_index,
        message=message,
        active_call_id=call_id or prior.active_call_id,
    )


class ChildLifecycleReducer:
    """Stateful reducer folding ACP collaboration and activity observations."""

    def __init__(self) -> None:
        self._children: dict[str, ChildLifecycleEntry] = {}
        self._malformed = False
        self._invalid_indexes: set[int] = set()

    @property
    def malformed(self) -> bool:
        return self._malformed

    @property
    def entries(self) -> dict[str, ChildLifecycleEntry]:
        return dict(self._children)

    @property
    def invalid_indexes(self) -> frozenset[int]:
        return frozenset(self._invalid_indexes)

    @property
    def transient_indexes(self) -> frozenset[int]:
        return frozenset(
            entry.last_index
            for entry in self._children.values()
            if entry.status in TRANSIENT_CHILD_STATUSES and entry.last_index >= 0
        )

    @property
    def has_transient(self) -> bool:
        return any(entry.is_transient for entry in self._children.values())

    def all_terminal(self) -> bool:
        return (
            not self._malformed
            and bool(self._children)
            and all(entry.is_terminal for entry in self._children.values())
        )

    def apply(
        self,
        tool_call: object,
        *,
        index: int = -1,
        event_name: str = "",
    ) -> ParsedChildToolCall:
        parsed = parse_child_tool_call(tool_call, event_name=event_name)
        if parsed.malformed:
            self._malformed = True
            if index >= 0:
                self._invalid_indexes.add(index)

        reopen_sources: set[str] = set()
        if parsed.starts_new_generation:
            reopen_sources.update(parsed.receivers)
            reopen_sources.update(obs.source_id for obs in parsed.observations)
        elif parsed.action in CHILD_GENERATION_ACTIONS:
            reopen_sources.update(
                obs.source_id
                for obs in parsed.observations
                if obs.status in TRANSIENT_CHILD_STATUSES
            )
        elif parsed.activity_is_transient and parsed.activity_source is not None:
            reopen_sources.add(parsed.activity_source)
        elif parsed.reopens_transient:
            reopen_sources.update(
                obs.source_id
                for obs in parsed.observations
                if obs.status in TRANSIENT_CHILD_STATUSES
            )

        for obs in parsed.observations:
            if obs.malformed or obs.status is None:
                self._malformed = True
                if index >= 0:
                    self._invalid_indexes.add(index)
            prior = self._children.get(obs.source_id)
            should_reopen = obs.source_id in reopen_sources and (
                parsed.starts_new_generation
                or obs.status in TRANSIENT_CHILD_STATUSES
                or (
                    parsed.activity_is_transient
                    and obs.source_id == parsed.activity_source
                )
            )
            self._children[obs.source_id] = reduce_child_entry(
                prior,
                source_id=obs.source_id,
                status=obs.status,
                raw_status=obs.raw_status,
                provenance=obs.provenance,
                malformed=obs.malformed,
                reopen=should_reopen,
                reopen_while_transient=(
                    parsed.action in CHILD_GENERATION_ACTIONS
                ),
                call_id=parsed.call_id,
                index=index,
                message=obs.message,
            )

        if parsed.failed_generation_with_activity and parsed.activity_source:
            prior = self._children.get(parsed.activity_source)
            self._children[parsed.activity_source] = reduce_child_entry(
                prior,
                source_id=parsed.activity_source,
                status="running",
                raw_status="running",
                provenance=ChildObservationProvenance.SUBAGENT_ACTIVITY,
                reopen=True,
                call_id=parsed.call_id,
                index=index,
            )

        return parsed


__all__ = [
    "CHILD_GENERATION_ACTIONS",
    "ChildLifecycleEntry",
    "ChildLifecycleReducer",
    "ChildObservationProvenance",
    "KNOWN_CHILD_ACTIONS",
    "KNOWN_CHILD_ACTIVITIES",
    "KNOWN_CHILD_STATUSES",
    "PASSIVE_CHILD_ACTIONS",
    "ParsedChildObservation",
    "ParsedChildToolCall",
    "TERMINAL_CHILD_STATUSES",
    "TRANSIENT_CHILD_STATUSES",
    "child_action_key",
    "collaboration_action_name",
    "iter_receiver_items",
    "iter_state_items",
    "normalize_provider_child_status",
    "parse_child_tool_call",
    "reduce_child_entry",
    "strict_source_id",
]
