import copy
import os
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional

from ..mode.backends import ALL_BACKEND_METAS
from ..utils.lock_order import LockLevel, ordered_lock

# Sentinel returned by add_chat_id() when the chat cannot be added
# (capacity full and all entries are owner).  Callers must check for
# this value to distinguish rejection from normal eviction (str) or
# no-eviction (None).
ADD_CHAT_ID_REJECTED = "__REJECTED__"


class ProjectStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass
class SessionSnapshot:
    session_id: str
    query_count: int
    last_query: str
    is_resumable: bool = True


_MAX_WORKFLOW_PENDING_REQUIREMENTS = 32
_MAX_WORKFLOW_PENDING_REQ_CHARS = 8192


def _sanitize_runtime_setting_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively sanitize runtime setting values into UTF-8-safe JSON primitives."""
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k or "").strip()
            if not key:
                continue
            out[key] = _sanitize_runtime_setting_value(v, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_runtime_setting_value(item, depth=depth + 1) for item in value[:64]]
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_runtime_settings_dict(raw: Any) -> dict[str, Any]:
    """Normalize a mapping of per-tool runtime settings."""
    if not isinstance(raw, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for raw_key, raw_val in raw.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        cleaned[key] = _sanitize_runtime_setting_value(raw_val)
    return cleaned


@dataclass
class ChatProjectState:
    """Per-chat isolated ACP tool/model configuration, mode flags, and snapshots."""

    acp_tool_name: Optional[str] = None
    acp_model_name: Optional[str] = None
    has_acp_config: bool = False
    active_modes: set[str] = field(default_factory=set)
    session_snapshots: dict[str, Optional[SessionSnapshot]] = field(default_factory=dict)
    workflow_team_config: Optional[dict[str, Any]] = None
    workflow_pending_requirements: OrderedDict[str, str] = field(default_factory=OrderedDict)
    tool_runtime_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _normalize_tool_runtime_settings_map(raw: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for raw_tool, raw_cfg in raw.items():
            tool_key = str(raw_tool or "").strip().lower().replace("-", "_")
            if not tool_key or not isinstance(raw_cfg, Mapping):
                continue
            cleaned = _sanitize_runtime_settings_dict(raw_cfg)
            if cleaned:
                normalized[tool_key] = cleaned
        return normalized

    def clone(self) -> "ChatProjectState":
        return ChatProjectState(
            acp_tool_name=self.acp_tool_name,
            acp_model_name=self.acp_model_name,
            has_acp_config=self.has_acp_config,
            active_modes=set(self.active_modes),
            session_snapshots={
                mode: (
                    SessionSnapshot(
                        session_id=snap.session_id,
                        query_count=snap.query_count,
                        last_query=snap.last_query,
                        is_resumable=snap.is_resumable,
                    )
                    if snap is not None
                    else None
                )
                for mode, snap in self.session_snapshots.items()
            },
            workflow_team_config=copy.deepcopy(self.workflow_team_config),
            workflow_pending_requirements=OrderedDict(self.workflow_pending_requirements),
            tool_runtime_settings={
                tool_key: copy.deepcopy(cfg)
                for tool_key, cfg in self.tool_runtime_settings.items()
                if isinstance(cfg, dict)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize per-chat project state to a snapshot dictionary."""
        payload: dict[str, Any] = {
            "acp_tool_name": self.acp_tool_name,
            "acp_model_name": self.acp_model_name,
            "has_acp_config": self.has_acp_config,
            "active_modes": sorted(self.active_modes),
            "session_snapshots": {
                mode_key: {
                    "session_id": snap.session_id,
                    "query_count": snap.query_count,
                    "last_query": snap.last_query,
                    "is_resumable": snap.is_resumable,
                }
                for mode_key, snap in self.session_snapshots.items()
                if snap is not None
            },
        }
        if self.workflow_team_config is not None:
            payload["workflow_team_config"] = copy.deepcopy(self.workflow_team_config)
        if self.workflow_pending_requirements:
            payload["workflow_pending_requirements"] = dict(self.workflow_pending_requirements)
        if self.tool_runtime_settings:
            payload["tool_runtime_settings"] = copy.deepcopy(self.tool_runtime_settings)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ChatProjectState":
        """Deserialize a ``ChatProjectState`` from a snapshot mapping safely."""
        if not isinstance(data, Mapping):
            return cls()
        raw_tool = data.get("acp_tool_name")
        acp_tool = str(raw_tool).strip() if isinstance(raw_tool, str) and raw_tool.strip() else None
        raw_model = data.get("acp_model_name")
        acp_model = str(raw_model) if isinstance(raw_model, str) and acp_tool else None
        has_acp_cfg = bool(data.get("has_acp_config", acp_tool is not None))
        active_modes = {
            str(m)
            for m in (data.get("active_modes") or ())
            if isinstance(m, str) and m.strip()
        }
        snapshots: dict[str, Optional[SessionSnapshot]] = {}
        raw_snaps = data.get("session_snapshots")
        if isinstance(raw_snaps, Mapping):
            for mode_key, sdata in raw_snaps.items():
                if isinstance(mode_key, str) and isinstance(sdata, Mapping) and "session_id" in sdata:
                    snapshots[mode_key] = SessionSnapshot(
                        session_id=str(sdata["session_id"]),
                        query_count=int(sdata.get("query_count", 0) or 0),
                        last_query=str(sdata.get("last_query", "") or ""),
                        is_resumable=bool(sdata.get("is_resumable", True)),
                    )
        raw_wf_cfg = data.get("workflow_team_config")
        wf_cfg = copy.deepcopy(dict(raw_wf_cfg)) if isinstance(raw_wf_cfg, Mapping) else None
        wf_pending: OrderedDict[str, str] = OrderedDict()
        raw_pending = data.get("workflow_pending_requirements")
        if isinstance(raw_pending, Mapping):
            for sid, req in raw_pending.items():
                if isinstance(sid, str) and isinstance(req, str) and sid.strip() and req.strip():
                    wf_pending[sid.strip()] = req.strip()[:_MAX_WORKFLOW_PENDING_REQ_CHARS]
        return cls(
            acp_tool_name=acp_tool,
            acp_model_name=acp_model,
            has_acp_config=has_acp_cfg,
            active_modes=active_modes,
            session_snapshots=snapshots,
            workflow_team_config=wf_cfg,
            workflow_pending_requirements=wf_pending,
            tool_runtime_settings=cls._normalize_tool_runtime_settings_map(
                data.get("tool_runtime_settings")
            ),
        )




@dataclass
class ProjectContext:
    project_id: str
    project_name: str
    root_path: str
    working_dir: str = ""
    status: ProjectStatus = ProjectStatus.IDLE
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    _active_modes: set[str] = field(default_factory=set, repr=False)
    session_snapshots: dict[str, Optional[SessionSnapshot]] = field(default_factory=dict, repr=False)
    chat_states: dict[str, ChatProjectState] = field(default_factory=dict, repr=False)
    _default_tool_runtime_settings: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    acp_tool_name: Optional[str] = None
    acp_model_name: Optional[str] = None

    theme_color: str = "green"
    emoji_prefix: str = "🟢"

    env_vars: dict = field(default_factory=dict)

    # ── Chat isolation fields ──
    # OrderedDict[chat_id, timestamp] ordered by last-access time (oldest first).
    # True LRU: move_to_end() on re-access (O(1)), evict via popitem(last=False).
    # Supports O(1) membership checks (``chat_id in allowed_chat_ids``).
    owner_chat_id: str = ""
    allowed_chat_ids: OrderedDict[str, float] = field(default_factory=OrderedDict)
    # In-memory only (not serialized): tracks chat_ids evicted by LRU for hint disambiguation.
    # Bounded OrderedDict — oldest entries are pruned when capacity exceeds max_evicted_cache.
    evicted_chat_ids: OrderedDict[str, float] = field(default_factory=OrderedDict, repr=False)

    # /new-chat 功能：项目专属群绑定
    bound_chat_id: str = ""
    bound_chat_name: str = ""
    bound_chat_created_at: float = 0.0
    managed_binding_generation: int = 0

    def __post_init__(self):
        # Lightweight lock protecting add_chat_id() mutations on
        # allowed_chat_ids and evicted_chat_ids.  Callers via
        # ProjectManager already hold _lock (RLock) — acquisition order
        # is always RLock → _chat_lock to avoid deadlocks.
        self._chat_lock = ordered_lock(LockLevel.CHAT_LOCK_CTX, name="ProjectContext._chat_lock")
        # Backward-compat: coerce list[tuple] from legacy callers/tests.
        # New code should pass OrderedDict directly (see create_project).
        if isinstance(self.allowed_chat_ids, list):
            self.allowed_chat_ids = OrderedDict(self.allowed_chat_ids)
        if not self.working_dir:
            self.working_dir = self.root_path
        self.root_path = os.path.expanduser(self.root_path)
        self.working_dir = os.path.expanduser(self.working_dir)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name not in {"acp_tool_name", "acp_model_name"}:
            return
        chat_states = self.__dict__.get("chat_states")
        if not isinstance(chat_states, dict) or len(chat_states) != 1:
            return
        sole_cid = next(iter(chat_states))
        if not sole_cid or self._known_other_chats_unlocked(sole_cid):
            return
        state = chat_states[sole_cid]
        if name == "acp_tool_name":
            state.acp_tool_name = value  # type: ignore[assignment]
            if value is not None:
                state.has_acp_config = True
        else:
            state.acp_model_name = value  # type: ignore[assignment]
            if self.__dict__.get("acp_tool_name") is not None or value is not None:
                state.has_acp_config = True

    @staticmethod
    def _normalized_chat_id(chat_id: Optional[str]) -> str:
        return str(chat_id or "").strip()

    def _known_other_chats_unlocked(self, chat_id: str) -> bool:
        chat_states = self.__dict__.get("chat_states")
        if isinstance(chat_states, dict) and any(
            cid and cid != chat_id for cid in chat_states
        ):
            return True
        allowed = self.__dict__.get("allowed_chat_ids")
        if allowed and any(cid and cid != chat_id for cid in allowed):
            return True
        for field_name in ("owner_chat_id", "bound_chat_id"):
            cid = str(self.__dict__.get(field_name) or "").strip()
            if cid and cid != chat_id:
                return True
        return False

    def _should_sync_legacy_unlocked(self, chat_id: str) -> bool:
        return (not chat_id) or (not self._known_other_chats_unlocked(chat_id))

    def _ensure_chat_state_unlocked(self, chat_id: str) -> ChatProjectState:
        state = self.chat_states.get(chat_id)
        if state is None:
            state = ChatProjectState()
            self.chat_states[chat_id] = state
        return state

    def _sync_sole_legacy_mode_attr(self, mode_name: str, enabled: bool) -> None:
        chat_states = self.__dict__.get("chat_states")
        if not isinstance(chat_states, dict) or len(chat_states) != 1:
            return
        sole_cid = next(iter(chat_states))
        if not sole_cid or self._known_other_chats_unlocked(sole_cid):
            return
        state = chat_states[sole_cid]
        if enabled:
            state.active_modes.add(mode_name)
        else:
            state.active_modes.discard(mode_name)

    def _sync_sole_legacy_snapshot_attr(
        self,
        mode_name: str,
        snap: Optional[SessionSnapshot],
    ) -> None:
        chat_states = self.__dict__.get("chat_states")
        if not isinstance(chat_states, dict) or len(chat_states) != 1:
            return
        sole_cid = next(iter(chat_states))
        if not sole_cid or self._known_other_chats_unlocked(sole_cid):
            return
        state = chat_states[sole_cid]
        if snap is not None:
            state.session_snapshots[mode_name] = snap
        else:
            state.session_snapshots.pop(mode_name, None)

    def get_acp_tool_name(self, chat_id: Optional[str] = None) -> Optional[str]:
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self.chat_states.get(cid)
                if state is not None and state.has_acp_config:
                    return state.acp_tool_name
            return self.acp_tool_name

    def get_acp_model_name(
        self,
        chat_id: Optional[str] = None,
        *,
        tool_name: Optional[str] = None,
    ) -> Optional[str]:
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            state = self.chat_states.get(cid) if cid else None
            if state is not None and state.has_acp_config:
                resolved_tool = state.acp_tool_name
                resolved_model = state.acp_model_name
            else:
                resolved_tool = self.acp_tool_name
                resolved_model = self.acp_model_name
            if tool_name is not None and resolved_tool != tool_name:
                return None
            return resolved_model

    def get_acp_config(
        self,
        chat_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            state = self.chat_states.get(cid) if cid else None
            if state is not None and state.has_acp_config:
                return state.acp_tool_name, state.acp_model_name
            return self.acp_tool_name, self.acp_model_name

    @staticmethod
    def _normalized_tool_runtime_key(tool_name: str) -> str:
        return str(tool_name or "").strip().lower().replace("-", "_")

    def get_tool_runtime_settings(
        self,
        chat_id: Optional[str],
        tool_name: str,
    ) -> dict[str, Any]:
        """Return a deep copy of per-chat runtime settings for *tool_name*."""
        tool_key = self._normalized_tool_runtime_key(tool_name)
        if not tool_key:
            return {}
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self.chat_states.get(cid)
                if state is None:
                    return {}
                cfg = state.tool_runtime_settings.get(tool_key)
                return copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
            cfg = self._default_tool_runtime_settings.get(tool_key)
            return copy.deepcopy(cfg) if isinstance(cfg, dict) else {}

    def set_tool_runtime_settings(
        self,
        chat_id: Optional[str],
        tool_name: str,
        settings: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Set per-chat runtime settings for *tool_name* under ``_chat_lock``."""
        tool_key = self._normalized_tool_runtime_key(tool_name)
        if not tool_key:
            return {}
        cid = self._normalized_chat_id(chat_id)
        cleaned = _sanitize_runtime_settings_dict(settings) if settings is not None else {}
        with self._chat_lock:
            if cid:
                if cleaned:
                    state = self._ensure_chat_state_unlocked(cid)
                    state.tool_runtime_settings[tool_key] = copy.deepcopy(cleaned)
                else:
                    existing_state = self.chat_states.get(cid)
                    if existing_state is not None:
                        existing_state.tool_runtime_settings.pop(tool_key, None)
            else:
                if cleaned:
                    self._default_tool_runtime_settings[tool_key] = copy.deepcopy(cleaned)
                else:
                    self._default_tool_runtime_settings.pop(tool_key, None)
            self.touch()
        return copy.deepcopy(cleaned)

    def get_workflow_team_config(
        self,
        chat_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Return a deep copy of the per-chat saved Workflow team config, if any."""
        cid = self._normalized_chat_id(chat_id)
        if not cid:
            return None
        with self._chat_lock:
            state = self.chat_states.get(cid)
            if state is None or state.workflow_team_config is None:
                return None
            return copy.deepcopy(state.workflow_team_config)

    def get_workflow_pending_requirement(
        self,
        config_session_id: str,
        chat_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return the per-chat pending `/wf_config <requirement>` string for *config_session_id*."""
        cid = self._normalized_chat_id(chat_id)
        sid = str(config_session_id or "").strip()
        if not cid or not sid:
            return None
        with self._chat_lock:
            state = self.chat_states.get(cid)
            if state is None:
                return None
            req = state.workflow_pending_requirements.get(sid)
            return req if isinstance(req, str) and req else None

    def get_programming_mode(
        self,
        mode_type: str,
        chat_id: Optional[str] = None,
    ) -> bool:
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self.chat_states.get(cid)
                if state is not None:
                    return mode_type in state.active_modes
                if self.chat_states:
                    return False
            return mode_type in self._active_modes

    def get_programming_snapshot(
        self,
        mode_type: str,
        chat_id: Optional[str] = None,
    ) -> Optional[SessionSnapshot]:
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self.chat_states.get(cid)
                if state is not None:
                    return state.session_snapshots.get(mode_type)
                if self.chat_states:
                    return None
            return self.session_snapshots.get(mode_type)

    def set_acp_configuration(
        self,
        tool_name: Optional[str],
        model_name: Optional[str],
        *,
        chat_id: Optional[str] = None,
    ) -> None:
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self._ensure_chat_state_unlocked(cid)
                state.acp_tool_name = tool_name
                state.acp_model_name = model_name
                state.has_acp_config = True
            if self._should_sync_legacy_unlocked(cid):
                object.__setattr__(self, "acp_tool_name", tool_name)
                object.__setattr__(self, "acp_model_name", model_name)

    def touch(self):
        self.last_active = time.time()

    def managed_group_migration_candidate(
        self,
        *,
        owner_id: str,
        receiving_bot_ref: str,
    ) -> dict[str, object] | None:
        """Return legacy binding facts that still require external validation.

        The candidate is deliberately inert: callers must pass it through the
        Registry's membership/receiving-bot validator before it can establish
        trust.  ``allowed_chat_ids`` and the bound group name are excluded.
        """

        if (
            not self.bound_chat_id
            or self.bound_chat_created_at <= 0
            or not self.root_path
            or not owner_id
            or not receiving_bot_ref
        ):
            return None
        return {
            "bound_chat_created_at": self.bound_chat_created_at,
            "canonical_root_ref": self.root_path,
            "chat_id": self.bound_chat_id,
            "origin": "ghostap_created",
            "owner_id": owner_id,
            "project_id": self.project_id,
            "receiving_bot_ref": receiving_bot_ref,
        }

    def _chat_id_set(self) -> frozenset[str]:
        """Return a frozenset of chat_ids — **test-only helper**.

        Hot-path callers should use ``chat_id in self.allowed_chat_ids``
        directly for O(1) lookup with zero allocation.  This method is
        retained solely for test assertions that compare against plain
        ``set`` / ``frozenset`` literals.
        """
        return frozenset(self.allowed_chat_ids)

    def add_chat_id(self, chat_id: str) -> Optional[str]:
        """Add *chat_id* to *allowed_chat_ids* with true LRU semantics.

        - If *chat_id* already exists, move it to the end (refresh timestamp).
        - If at capacity, evict the oldest non-owner entry.
        - Returns the evicted chat_id if eviction occurred, else ``None``.
        - Returns :data:`ADD_CHAT_ID_REJECTED` when the chat cannot be added
          (capacity full and all entries are owner).  Callers **must** check
          for this sentinel to detect rejection.
        """
        with self._chat_lock:
            now = time.time()
            # Move-to-end if already present — O(1)
            if chat_id in self.allowed_chat_ids:
                self.allowed_chat_ids[chat_id] = now
                self.allowed_chat_ids.move_to_end(chat_id)
                return None

            from ..config import get_settings
            limit = get_settings().max_allowed_chat_ids
            evicted_chat_id: Optional[str] = None

            # Evict oldest non-owner entry when at capacity
            _eviction_failed = False
            while len(self.allowed_chat_ids) >= limit:
                # Iterate from oldest (front) to find a non-owner entry
                victim_cid: Optional[str] = None
                for cid in self.allowed_chat_ids:
                    if cid != self.owner_chat_id:
                        victim_cid = cid
                        break
                if victim_cid is None:
                    # All entries are owner — cannot evict without violating invariant
                    _eviction_failed = True
                    break
                evicted_chat_id = victim_cid
                del self.allowed_chat_ids[victim_cid]
                self.chat_states.pop(victim_cid, None)
                # Record eviction in bounded OrderedDict (LRU pruning)
                self.evicted_chat_ids[victim_cid] = now
                self.evicted_chat_ids.move_to_end(victim_cid)
                try:
                    _evicted_limit = int(get_settings().max_evicted_cache)
                except (TypeError, ValueError, AttributeError):
                    _evicted_limit = 200
                while len(self.evicted_chat_ids) > _evicted_limit:
                    self.evicted_chat_ids.popitem(last=False)

            if _eviction_failed and len(self.allowed_chat_ids) >= limit:
                import logging
                logging.getLogger(__name__).warning(
                    "add_chat_id: cannot evict (all entries are owner), "
                    "rejecting chat_id=%s for project=%s (limit=%d)",
                    chat_id[:12] if chat_id else chat_id,
                    self.project_id,
                    limit,
                )
                return ADD_CHAT_ID_REJECTED

            self.allowed_chat_ids[chat_id] = now
            return evicted_chat_id

    def clear_session_state(
        self,
        mode_type: Optional[str] = None,
        *,
        chat_id: Optional[str] = None,
    ) -> None:
        """Clear active snapshot while keeping mode/tool/model config."""
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if mode_type and mode_type in self._MODE_ATTRS:
                if cid:
                    state = self._ensure_chat_state_unlocked(cid)
                    state.session_snapshots.pop(mode_type, None)
                if self._should_sync_legacy_unlocked(cid):
                    self.session_snapshots.pop(mode_type, None)
            self.touch()

    # ── Mode name → (mode_flag_attr, snapshot_attr) mapping ──
    _MODE_ATTRS: ClassVar[dict[str, tuple[str, str]]] = {
        spec.mode_key: (spec.project_mode_attr, spec.project_snapshot_attr)
        for spec in ALL_BACKEND_METAS
    }

    def set_programming_mode(
        self,
        mode_type: str,
        enabled: bool,
        session_id: Optional[str] = None,
        query_count: int = 0,
        *,
        chat_id: Optional[str] = None,
    ):
        """Generic mode setter — replaces per-mode set_*_mode methods."""
        if mode_type not in self._MODE_ATTRS:
            raise KeyError(mode_type)
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self._ensure_chat_state_unlocked(cid)
                if enabled:
                    state.active_modes.add(mode_type)
                    if session_id:
                        state.session_snapshots[mode_type] = SessionSnapshot(
                            session_id=session_id,
                            query_count=query_count,
                            last_query="",
                            is_resumable=True,
                        )
                else:
                    state.active_modes.discard(mode_type)
                    snap = state.session_snapshots.get(mode_type)
                    if snap:
                        snap.is_resumable = True
            if self._should_sync_legacy_unlocked(cid):
                if enabled:
                    self._active_modes.add(mode_type)
                    if session_id:
                        self.session_snapshots[mode_type] = SessionSnapshot(
                            session_id=session_id,
                            query_count=query_count,
                            last_query="",
                            is_resumable=True,
                        )
                else:
                    self._active_modes.discard(mode_type)
                    snap = self.session_snapshots.get(mode_type)
                    if snap:
                        snap.is_resumable = True

    def update_programming_snapshot(
        self,
        mode_type: str,
        query: str,
        query_count: int,
        session_id: Optional[str] = None,
        *,
        chat_id: Optional[str] = None,
    ):
        """Generic snapshot updater — replaces per-mode update_*_snapshot methods."""
        if mode_type not in self._MODE_ATTRS:
            raise KeyError(mode_type)
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            if cid:
                state = self._ensure_chat_state_unlocked(cid)
                snap = state.session_snapshots.get(mode_type)
                if snap is not None:
                    snap.last_query = query
                    snap.query_count = query_count
                    if session_id:
                        snap.session_id = session_id
                elif session_id:
                    state.session_snapshots[mode_type] = SessionSnapshot(
                        session_id=session_id,
                        query_count=query_count,
                        last_query=query,
                        is_resumable=True,
                    )
            if self._should_sync_legacy_unlocked(cid):
                snap = self.session_snapshots.get(mode_type)
                if snap is not None:
                    snap.last_query = query
                    snap.query_count = query_count
                    if session_id:
                        snap.session_id = session_id

    def clear_programming_snapshot_if_matches(
        self,
        mode_type: str,
        session_id: str,
        *,
        chat_id: Optional[str] = None,
    ) -> bool:
        """Compare-and-clear a snapshot without erasing a newer replacement."""
        expected = str(session_id or "")
        if not expected or mode_type not in self._MODE_ATTRS:
            return False
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            cleared = False
            if cid:
                state = self.chat_states.get(cid)
                snapshot = state.session_snapshots.get(mode_type) if state else None
                if str(getattr(snapshot, "session_id", "") or "") == expected:
                    state.session_snapshots.pop(mode_type, None)  # type: ignore[union-attr]
                    cleared = True
            if self._should_sync_legacy_unlocked(cid):
                snapshot = self.session_snapshots.get(mode_type)
                if str(getattr(snapshot, "session_id", "") or "") == expected:
                    self.session_snapshots.pop(mode_type, None)
                    cleared = True
            return cleared

    def commit_acp_programming_activation(
        self,
        mode_type: str,
        tool_name: str,
        model_name: Optional[str],
        session_id: str,
        query_count: int = 0,
        activate_mode: bool = True,
        *,
        chat_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Atomically select an ACP backend and activate its project mode.

        The returned state is private rollback data for ``ProjectManager`` if
        persistence fails.  Callers must hold the manager lock before invoking
        this method so a persisted activation cannot interleave with another
        project-wide commit.
        """
        if mode_type not in self._MODE_ATTRS:
            raise ValueError(f"unsupported programming mode: {mode_type}")
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            existing_state = self.chat_states.get(cid) if cid else None
            previous: dict[str, object] = {
                "chat_id": cid,
                "had_chat_state": existing_state is not None,
                "chat_state": existing_state.clone() if existing_state is not None else None,
                "acp_tool_name": self.acp_tool_name,
                "acp_model_name": self.acp_model_name,
                "last_active": self.last_active,
                "modes": {
                    name: (
                        name in self._active_modes,
                        self.session_snapshots.get(name),
                    )
                    for name in self._MODE_ATTRS
                },
            }
            new_snap = SessionSnapshot(
                session_id=session_id,
                query_count=query_count,
                last_query="",
                is_resumable=True,
            )
            if cid:
                state = self._ensure_chat_state_unlocked(cid)
                if activate_mode:
                    state.active_modes.clear()
                    state.active_modes.add(mode_type)
                    state.session_snapshots[mode_type] = SessionSnapshot(
                        session_id=new_snap.session_id,
                        query_count=new_snap.query_count,
                        last_query=new_snap.last_query,
                        is_resumable=new_snap.is_resumable,
                    )
                state.acp_tool_name = tool_name
                state.acp_model_name = model_name
                state.has_acp_config = True
            if self._should_sync_legacy_unlocked(cid):
                if activate_mode:
                    self._active_modes.clear()
                    self._active_modes.add(mode_type)
                    self.session_snapshots[mode_type] = new_snap
                object.__setattr__(self, "acp_tool_name", tool_name)
                object.__setattr__(self, "acp_model_name", model_name)
            self.touch()
            return previous

    def replace_acp_configuration(
        self,
        tool_name: str,
        model_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Replace the saved tool/model pair and return legacy 2-tuple rollback state."""
        with self._chat_lock:
            previous = (self.acp_tool_name, self.acp_model_name)
            self.acp_tool_name = tool_name
            self.acp_model_name = model_name
            return previous

    def replace_acp_configuration_for_chat(
        self,
        tool_name: str,
        model_name: Optional[str],
        *,
        chat_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Replace saved tool/model configuration scoped to *chat_id* with full rollback."""
        cid = self._normalized_chat_id(chat_id)
        with self._chat_lock:
            existing_state = self.chat_states.get(cid) if cid else None
            previous: dict[str, object] = {
                "chat_id": cid,
                "had_chat_state": existing_state is not None,
                "chat_state": existing_state.clone() if existing_state is not None else None,
                "acp_tool_name": self.acp_tool_name,
                "acp_model_name": self.acp_model_name,
            }
            if cid:
                state = self._ensure_chat_state_unlocked(cid)
                state.acp_tool_name = tool_name
                state.acp_model_name = model_name
                state.has_acp_config = True
            if self._should_sync_legacy_unlocked(cid):
                object.__setattr__(self, "acp_tool_name", tool_name)
                object.__setattr__(self, "acp_model_name", model_name)
            return previous

    def restore_acp_configuration(self, previous: dict[str, object]) -> None:
        """Restore rollback data from a failed persisted ACP configuration change."""
        with self._chat_lock:
            cid = self._normalized_chat_id(previous.get("chat_id"))  # type: ignore[arg-type]
            if cid:
                if previous.get("had_chat_state"):
                    prev_state = previous.get("chat_state")
                    if isinstance(prev_state, ChatProjectState):
                        self.chat_states[cid] = prev_state.clone()
                else:
                    self.chat_states.pop(cid, None)
            object.__setattr__(self, "acp_tool_name", previous["acp_tool_name"])
            object.__setattr__(self, "acp_model_name", previous["acp_model_name"])

    def replace_workflow_team_config_for_chat(
        self,
        config: Optional[dict[str, Any]],
        *,
        chat_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Replace saved Workflow team config scoped to *chat_id* with full rollback."""
        cid = self._normalized_chat_id(chat_id)
        normalized: Optional[dict[str, Any]]
        if config is None:
            normalized = None
        else:
            normalized = self.normalize_workflow_team_config(config)
            if normalized is None:
                raise ValueError("Invalid workflow_team_config")
        with self._chat_lock:
            existing_state = self.chat_states.get(cid) if cid else None
            previous: dict[str, object] = {
                "chat_id": cid,
                "had_chat_state": existing_state is not None,
                "chat_state": existing_state.clone() if existing_state is not None else None,
            }
            if cid:
                state = self._ensure_chat_state_unlocked(cid)
                state.workflow_team_config = copy.deepcopy(normalized)
            return previous

    def restore_workflow_team_config(self, previous: dict[str, object]) -> None:
        """Restore rollback data from a failed persisted Workflow team config change."""
        with self._chat_lock:
            cid = self._normalized_chat_id(previous.get("chat_id"))  # type: ignore[arg-type]
            if cid:
                if previous.get("had_chat_state"):
                    prev_state = previous.get("chat_state")
                    if isinstance(prev_state, ChatProjectState):
                        self.chat_states[cid] = prev_state.clone()
                else:
                    self.chat_states.pop(cid, None)

    def replace_workflow_pending_requirement_for_chat(
        self,
        config_session_id: str,
        requirement: Optional[str],
        *,
        chat_id: Optional[str] = None,
    ) -> dict[str, object]:
        """Set or clear a bounded pending `/wf_config` requirement scoped to *chat_id*."""
        cid = self._normalized_chat_id(chat_id)
        sid = str(config_session_id or "").strip()
        req = (
            requirement.strip()[:_MAX_WORKFLOW_PENDING_REQ_CHARS]
            if isinstance(requirement, str) and requirement.strip()
            else None
        )
        with self._chat_lock:
            existing_state = self.chat_states.get(cid) if cid else None
            previous: dict[str, object] = {
                "chat_id": cid,
                "had_chat_state": existing_state is not None,
                "chat_state": existing_state.clone() if existing_state is not None else None,
            }
            if cid and sid and len(sid) <= 128:
                if req is not None:
                    state = self._ensure_chat_state_unlocked(cid)
                    state.workflow_pending_requirements[sid] = req
                    state.workflow_pending_requirements.move_to_end(sid)
                    while len(state.workflow_pending_requirements) > _MAX_WORKFLOW_PENDING_REQUIREMENTS:
                        state.workflow_pending_requirements.popitem(last=False)
                elif existing_state is not None:
                    existing_state.workflow_pending_requirements.pop(sid, None)
            return previous

    def restore_workflow_pending_requirement(self, previous: dict[str, object]) -> None:
        """Restore rollback data from a failed persisted pending-requirement mutation."""
        with self._chat_lock:
            cid = self._normalized_chat_id(previous.get("chat_id"))  # type: ignore[arg-type]
            if cid:
                if previous.get("had_chat_state"):
                    prev_state = previous.get("chat_state")
                    if isinstance(prev_state, ChatProjectState):
                        self.chat_states[cid] = prev_state.clone()
                else:
                    self.chat_states.pop(cid, None)

    def restore_acp_programming_activation(self, previous: dict[str, object]) -> None:
        """Restore rollback data from a failed persisted ACP activation."""
        with self._chat_lock:
            cid = self._normalized_chat_id(previous.get("chat_id"))  # type: ignore[arg-type]
            if cid:
                if previous.get("had_chat_state"):
                    prev_state = previous.get("chat_state")
                    if isinstance(prev_state, ChatProjectState):
                        self.chat_states[cid] = prev_state.clone()
                else:
                    self.chat_states.pop(cid, None)
            object.__setattr__(self, "acp_tool_name", previous["acp_tool_name"])
            object.__setattr__(self, "acp_model_name", previous["acp_model_name"])
            self.last_active = float(previous["last_active"])  # type: ignore[arg-type]
            modes = previous["modes"]
            for name in self._MODE_ATTRS:
                enabled, snapshot = modes[name]  # type: ignore[index]
                if enabled:
                    self._active_modes.add(name)
                else:
                    self._active_modes.discard(name)
                if snapshot is not None:
                    self.session_snapshots[name] = snapshot
                else:
                    self.session_snapshots.pop(name, None)

    def get_status_emoji(self) -> str:
        status_map = {
            ProjectStatus.IDLE: "⚪",
            ProjectStatus.ACTIVE: self.emoji_prefix,
            ProjectStatus.BUSY: "🟡",
            ProjectStatus.SUSPENDED: "⏸️",
            ProjectStatus.CLOSED: "❌",
        }
        return status_map.get(self.status, "⚪")

    @staticmethod
    def _snap_to_dict(snap: Optional[SessionSnapshot]) -> Optional[dict]:
        if snap is None:
            return None
        return {
            "session_id": snap.session_id,
            "query_count": snap.query_count,
            "last_query": snap.last_query,
            "is_resumable": snap.is_resumable,
        }

    def to_snapshot(self) -> dict:
        d: dict = {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "root_path": self.root_path,
            "working_dir": self.working_dir,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "acp_tool_name": self.acp_tool_name,
            "acp_model_name": self.acp_model_name,
            "theme_color": self.theme_color,
            "emoji_prefix": self.emoji_prefix,
            "env_vars": self.env_vars,
            "owner_chat_id": self.owner_chat_id,
            "allowed_chat_ids": [[cid, ts] for cid, ts in self.allowed_chat_ids.items()],
            "bound_chat_id": self.bound_chat_id,
            "bound_chat_name": self.bound_chat_name,
            "bound_chat_created_at": self.bound_chat_created_at,
            "managed_binding_generation": self.managed_binding_generation,
        }
        with self._chat_lock:
            for mode_type, (mode_flag, snap_attr) in self._MODE_ATTRS.items():
                d[mode_flag] = getattr(self, mode_flag)
                d[snap_attr] = self._snap_to_dict(getattr(self, snap_attr))
            if self._default_tool_runtime_settings:
                d["tool_runtime_settings"] = copy.deepcopy(self._default_tool_runtime_settings)
            if self.chat_states:
                d["chat_states"] = {
                    cid: st.to_dict()
                    for cid, st in self.chat_states.items()
                }
        return d

    @staticmethod
    def _parse_allowed_chat_ids(raw: list) -> OrderedDict[str, float]:
        """Parse allowed_chat_ids from snapshot with backward compatibility.

        Accepts two formats:
        - New: ``[[chat_id, timestamp], ...]``
        - Legacy: ``["chat_id_1", "chat_id_2", ...]`` (assigns incremental timestamps)
        """
        if not raw:
            return OrderedDict()
        first = raw[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            # New format: list of [chat_id, timestamp] pairs
            return OrderedDict((str(entry[0]), float(entry[1])) for entry in raw)
        # Legacy format: list of plain strings — assign incremental timestamps
        base_ts = time.time() - len(raw)
        return OrderedDict((str(entry), base_ts + i) for i, entry in enumerate(raw))

    @staticmethod
    def normalize_acp_tool_name(tool_name: object) -> Optional[str]:
        """Return a registered ACP tool name, or ``None`` for stale input."""
        name = str(tool_name or "").strip().lower()
        if not name:
            return None
        from ..acp.providers import get_providers, tool_registry

        get_providers()
        return name if tool_registry.get_provider(name) else None

    @classmethod
    def normalize_workflow_team_config(cls, raw: Any) -> Optional[dict[str, Any]]:
        """Validate and sanitize a persisted ``workflow_team_config`` payload.

        Corrupt, non-dict, stale-tool, or invalid-effort payloads recover cleanly
        to either a sanitized subset or ``None`` without ever raising.
        """
        if not isinstance(raw, dict):
            return None
        raw_bindings = raw.get("bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            return None

        try:
            from ..workflow_engine.agent_pool import (
                WorkflowAgentBinding,
                freeze_agent_pool,
            )
            from ..workflow_engine.constants import MAX_WORKFLOW_AGENT_POOL_SIZE
            from ..workflow_engine.teamwork import PRESET_ALIASES, TEAMWORK_PRESETS
            from ..workflow_engine.tool_registry import (
                get_tool_capabilities,
                validate_agent_binding_spec,
            )
        except Exception:
            return None

        raw_preset = (
            str(raw.get("preset_id") or raw.get("preset") or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        valid_presets = set(TEAMWORK_PRESETS) | {"auto", "custom"}
        if raw_preset in valid_presets:
            preset_id = raw_preset
        elif raw_preset in PRESET_ALIASES:
            preset_id = PRESET_ALIASES[raw_preset]
        else:
            preset_id = "recommended"

        try:
            capabilities = get_tool_capabilities()
        except Exception:
            capabilities = {}

        cleaned: list[WorkflowAgentBinding] = []
        seen_ids: set[str] = set()
        seen_keys: set[tuple[str, str, str, str]] = set()
        for item in raw_bindings[:64]:
            binding = WorkflowAgentBinding.from_dict(item)
            if binding is None:
                continue
            norm_tool = cls.normalize_acp_tool_name(binding.tool_name)
            if not norm_tool:
                continue

            candidate: WorkflowAgentBinding | None = None
            attempts = (
                (binding.model_name, binding.profile, binding.effort),
                (binding.model_name, None, None),
                (None, None, None),
            )
            for cand_model, cand_profile, cand_effort in attempts:
                ok, _ = validate_agent_binding_spec(
                    norm_tool,
                    model_name=cand_model,
                    profile=cand_profile,
                    effort=cand_effort,
                    available_tools={norm_tool: norm_tool},
                    capabilities=capabilities or None,
                )
                if ok:
                    candidate = WorkflowAgentBinding(
                        agent_id=binding.agent_id,
                        tool_name=norm_tool,
                        model_name=cand_model,
                        display_name=binding.display_name,
                        profile=cand_profile,
                        effort=cand_effort,
                        role=binding.role,
                        roles=binding.roles,
                    )
                    break

            if candidate is None:
                continue
            if candidate.agent_id in seen_ids or candidate.binding_key in seen_keys:
                continue
            seen_ids.add(candidate.agent_id)
            seen_keys.add(candidate.binding_key)
            cleaned.append(candidate)
            if len(cleaned) >= MAX_WORKFLOW_AGENT_POOL_SIZE:
                break

        if not cleaned:
            return None

        try:
            frozen = freeze_agent_pool(cleaned, require_nonempty=False)
        except ValueError:
            return None
        if not frozen:
            return None

        valid_ids = {b.agent_id for b in frozen}
        raw_orch = str(raw.get("orchestrator_agent_id") or "").strip()
        orch_id = raw_orch if raw_orch in valid_ids else frozen[0].agent_id

        raw_ts = raw.get("updated_at")
        updated_at = (
            float(raw_ts)
            if isinstance(raw_ts, (int, float)) and not isinstance(raw_ts, bool)
            else 0.0
        )
        return {
            "version": 1,
            "preset_id": preset_id,
            "orchestrator_agent_id": orch_id,
            "bindings": [b.to_dict() for b in frozen],
            "updated_at": updated_at,
        }

    @staticmethod
    def normalize_workflow_pending_requirements(raw: Any) -> OrderedDict[str, str]:
        """Sanitize and bound persisted `(config_session_id -> requirement)` entries."""
        out: OrderedDict[str, str] = OrderedDict()
        if not isinstance(raw, dict):
            return out
        for raw_sid, raw_req in raw.items():
            if not isinstance(raw_sid, str) or not isinstance(raw_req, str):
                continue
            sid = raw_sid.strip()
            req = raw_req.strip()
            if not sid or len(sid) > 128 or not req:
                continue
            out[sid] = req[:_MAX_WORKFLOW_PENDING_REQ_CHARS]
            out.move_to_end(sid)
            while len(out) > _MAX_WORKFLOW_PENDING_REQUIREMENTS:
                out.popitem(last=False)
        return out

    @classmethod
    def from_snapshot(cls, data: dict) -> "ProjectContext":
        acp_tool_name = cls.normalize_acp_tool_name(data.get("acp_tool_name"))
        ctx = cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            root_path=data["root_path"],
            working_dir=data.get("working_dir", data["root_path"]),
            status=ProjectStatus(data.get("status", "idle")),
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
            acp_tool_name=acp_tool_name,
            acp_model_name=data.get("acp_model_name") if acp_tool_name else None,
            theme_color=data.get("theme_color", "green"),
            emoji_prefix=data.get("emoji_prefix", "🟢"),
            env_vars=data.get("env_vars", {}),
            owner_chat_id=data.get("owner_chat_id", ""),
            allowed_chat_ids=cls._parse_allowed_chat_ids(data.get("allowed_chat_ids", [])),
            bound_chat_id=data.get("bound_chat_id", ""),
            bound_chat_name=data.get("bound_chat_name", ""),
            bound_chat_created_at=data.get("bound_chat_created_at", 0.0),
            managed_binding_generation=data.get("managed_binding_generation", 0),
        )
        for mode_flag, snap_attr in cls._MODE_ATTRS.values():
            if data.get(mode_flag, False):
                setattr(ctx, mode_flag, True)
            snap_data = data.get(snap_attr)
            if snap_data:
                setattr(ctx, snap_attr, SessionSnapshot(
                    session_id=snap_data["session_id"],
                    query_count=snap_data["query_count"],
                    last_query=snap_data["last_query"],
                    is_resumable=snap_data.get("is_resumable", True),
                ))
        raw_chat_states = data.get("chat_states")
        if isinstance(raw_chat_states, dict):
            for raw_cid, raw_state in raw_chat_states.items():
                cid = cls._normalized_chat_id(raw_cid)
                if not cid or not isinstance(raw_state, dict):
                    continue
                chat_tool = cls.normalize_acp_tool_name(raw_state.get("acp_tool_name"))
                raw_has_cfg = bool(raw_state.get("has_acp_config", chat_tool is not None))
                chat_model = (
                    raw_state.get("acp_model_name")
                    if chat_tool
                    else None
                )
                active_modes = {
                    str(m)
                    for m in (raw_state.get("active_modes") or ())
                    if str(m) in cls._MODE_ATTRS
                }
                snapshots: dict[str, Optional[SessionSnapshot]] = {}
                raw_snaps = raw_state.get("session_snapshots")
                if isinstance(raw_snaps, dict):
                    for mode_key, sdata in raw_snaps.items():
                        if mode_key in cls._MODE_ATTRS and isinstance(sdata, dict):
                            snapshots[mode_key] = SessionSnapshot(
                                session_id=sdata["session_id"],
                                query_count=sdata["query_count"],
                                last_query=sdata["last_query"],
                                is_resumable=sdata.get("is_resumable", True),
                            )
                try:
                    wf_team_cfg = cls.normalize_workflow_team_config(
                        raw_state.get("workflow_team_config")
                    )
                except Exception:
                    wf_team_cfg = None
                wf_pending_reqs = cls.normalize_workflow_pending_requirements(
                    raw_state.get("workflow_pending_requirements")
                )
                tool_runtime = ChatProjectState._normalize_tool_runtime_settings_map(
                    raw_state.get("tool_runtime_settings")
                )
                ctx.chat_states[cid] = ChatProjectState(
                    acp_tool_name=chat_tool,
                    acp_model_name=chat_model,
                    has_acp_config=raw_has_cfg,
                    active_modes=active_modes,
                    session_snapshots=snapshots,
                    workflow_team_config=wf_team_cfg,
                    workflow_pending_requirements=wf_pending_reqs,
                    tool_runtime_settings=tool_runtime,
                )
        ctx._default_tool_runtime_settings = ChatProjectState._normalize_tool_runtime_settings_map(
            data.get("tool_runtime_settings")
        )
        return ctx


def _set_legacy_mode_descriptor(
    project: ProjectContext,
    mode_name: str,
    enabled: bool,
) -> None:
    if enabled:
        project._active_modes.add(mode_name)
    else:
        project._active_modes.discard(mode_name)
    project._sync_sole_legacy_mode_attr(mode_name, enabled)


def _set_legacy_snapshot_descriptor(
    project: ProjectContext,
    mode_name: str,
    snap: Optional[SessionSnapshot],
) -> None:
    if snap is not None:
        project.session_snapshots[mode_name] = snap
    else:
        project.session_snapshots.pop(mode_name, None)
    project._sync_sole_legacy_snapshot_attr(mode_name, snap)


def _attach_mode_descriptors() -> None:
    """Attach property accessors for each programming mode flag and snapshot."""
    for mode_name, (mode_flag, snap_attr) in ProjectContext._MODE_ATTRS.items():
        setattr(
            ProjectContext,
            mode_flag,
            property(
                lambda self, m=mode_name: m in self._active_modes,
                lambda self, enabled, m=mode_name: _set_legacy_mode_descriptor(
                    self, m, bool(enabled)
                ),
            ),
        )
        setattr(
            ProjectContext,
            snap_attr,
            property(
                lambda self, m=mode_name: self.session_snapshots.get(m),
                lambda self, snap, m=mode_name: _set_legacy_snapshot_descriptor(
                    self, m, snap
                ),
            ),
        )


_attach_mode_descriptors()


def get_project_acp_tool_name(
    project: object,
    chat_id: Optional[str] = None,
) -> Optional[str]:
    """Read *project*'s ACP tool name scoped to *chat_id* when supported."""
    if project is None:
        return None
    if chat_id and callable(getattr(type(project), "get_acp_tool_name", None)):
        return project.get_acp_tool_name(chat_id)  # type: ignore[attr-defined]
    return getattr(project, "acp_tool_name", None)


def get_project_acp_model_name(
    project: object,
    chat_id: Optional[str] = None,
    *,
    tool_name: Optional[str] = None,
) -> Optional[str]:
    """Read *project*'s ACP model name scoped to *chat_id* and optional *tool_name*."""
    if project is None:
        return None
    if chat_id and callable(getattr(type(project), "get_acp_model_name", None)):
        return project.get_acp_model_name(chat_id, tool_name=tool_name)  # type: ignore[attr-defined]
    if tool_name is not None and getattr(project, "acp_tool_name", None) != tool_name:
        return None
    return getattr(project, "acp_model_name", None)


def get_project_programming_snapshot(
    project: object,
    mode_type: str,
    chat_id: Optional[str] = None,
) -> Optional[SessionSnapshot]:
    """Read *project*'s session snapshot for *mode_type* scoped to *chat_id*."""
    if project is None:
        return None
    if chat_id and callable(getattr(type(project), "get_programming_snapshot", None)):
        return project.get_programming_snapshot(mode_type, chat_id)  # type: ignore[attr-defined]
    pair = ProjectContext._MODE_ATTRS.get(mode_type)
    if pair is None:
        return None
    return getattr(project, pair[1], None)


def get_project_programming_mode(
    project: object,
    mode_type: str,
    chat_id: Optional[str] = None,
) -> bool:
    """Read whether *mode_type* is active on *project* for *chat_id*."""
    if project is None:
        return False
    if chat_id and callable(getattr(type(project), "get_programming_mode", None)):
        return bool(project.get_programming_mode(mode_type, chat_id))  # type: ignore[attr-defined]
    pair = ProjectContext._MODE_ATTRS.get(mode_type)
    if pair is None:
        return False
    return getattr(project, pair[0], False) is True


def get_project_workflow_team_config(
    project: object,
    chat_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Read *project*'s saved Workflow team configuration scoped to *chat_id*."""
    if project is None or not chat_id:
        return None
    getter = getattr(type(project), "get_workflow_team_config", None)
    if callable(getter):
        return project.get_workflow_team_config(chat_id)  # type: ignore[attr-defined]
    return None


def get_project_workflow_pending_requirement(
    project: object,
    config_session_id: str,
    chat_id: Optional[str] = None,
) -> Optional[str]:
    """Read *project*'s pending `/wf_config` requirement scoped to *chat_id* and *config_session_id*."""
    if project is None or not chat_id or not config_session_id:
        return None
    getter = getattr(type(project), "get_workflow_pending_requirement", None)
    if callable(getter):
        return project.get_workflow_pending_requirement(config_session_id, chat_id)  # type: ignore[attr-defined]
    return None
