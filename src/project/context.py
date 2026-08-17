import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from ..utils.lock_order import LockLevel, ordered_lock

if TYPE_CHECKING:
    pass

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
class ConversationItem:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: dict
    status: str = "pending"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class SessionSnapshot:
    session_id: str
    query_count: int
    last_query: str
    is_resumable: bool = True




@dataclass
class ProjectContext:
    project_id: str
    project_name: str
    root_path: str
    working_dir: str = ""
    status: ProjectStatus = ProjectStatus.IDLE
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    coco_session_snapshot: Optional[SessionSnapshot] = None
    coco_mode: bool = False

    claude_session_snapshot: Optional[SessionSnapshot] = None
    claude_mode: bool = False

    aiden_session_snapshot: Optional[SessionSnapshot] = None
    aiden_mode: bool = False

    codex_session_snapshot: Optional[SessionSnapshot] = None
    codex_mode: bool = False

    gemini_session_snapshot: Optional[SessionSnapshot] = None
    gemini_mode: bool = False

    traex_session_snapshot: Optional[SessionSnapshot] = None
    traex_mode: bool = False

    grok_session_snapshot: Optional[SessionSnapshot] = None
    grok_mode: bool = False

    dsh_session_snapshot: Optional[SessionSnapshot] = None
    dsh_mode: bool = False

    acp_tool_name: Optional[str] = None
    acp_model_name: Optional[str] = None

    task_queue: list[Task] = field(default_factory=list)
    current_task: Optional[Task] = None

    conversation_history: list[ConversationItem] = field(default_factory=list)
    max_history_size: int = 20

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

    def add_conversation(self, role: str, content: str, message_id: Optional[str] = None):
        item = ConversationItem(role=role, content=content, message_id=message_id)
        self.conversation_history.append(item)
        if len(self.conversation_history) > self.max_history_size:
            self.conversation_history = self.conversation_history[-self.max_history_size :]
        self.touch()

    # ── Mode name → (mode_flag_attr, snapshot_attr) mapping ──
    _MODE_ATTRS: ClassVar[dict[str, tuple[str, str]]] = {
        "coco": ("coco_mode", "coco_session_snapshot"),
        "claude": ("claude_mode", "claude_session_snapshot"),
        "aiden": ("aiden_mode", "aiden_session_snapshot"),
        "codex": ("codex_mode", "codex_session_snapshot"),
        "gemini": ("gemini_mode", "gemini_session_snapshot"),
        "traex": ("traex_mode", "traex_session_snapshot"),
        "grok": ("grok_mode", "grok_session_snapshot"),
        "dsh": ("dsh_mode", "dsh_session_snapshot"),
    }

    def set_programming_mode(self, mode_type: str, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        """Generic mode setter — replaces per-mode set_*_mode methods."""
        mode_flag, snap_attr = self._MODE_ATTRS[mode_type]
        with self._chat_lock:
            setattr(self, mode_flag, enabled)
            if enabled and session_id:
                setattr(self, snap_attr, SessionSnapshot(
                    session_id=session_id, query_count=query_count, last_query="", is_resumable=True
                ))
            elif not enabled:
                snap = getattr(self, snap_attr)
                if snap:
                    snap.is_resumable = True

    def update_programming_snapshot(self, mode_type: str, query: str, query_count: int, session_id: Optional[str] = None):
        """Generic snapshot updater — replaces per-mode update_*_snapshot methods."""
        _, snap_attr = self._MODE_ATTRS[mode_type]
        with self._chat_lock:
            snap = getattr(self, snap_attr)
            if snap:
                snap.last_query = query
                snap.query_count = query_count
                if session_id:
                    snap.session_id = session_id

    def get_programming_snapshot(
        self,
        mode_type: str,
    ) -> Optional[SessionSnapshot]:
        """Read one programming snapshot under the context mutation lock."""
        _, snap_attr = self._MODE_ATTRS[mode_type]
        with self._chat_lock:
            return getattr(self, snap_attr)

    def clear_programming_snapshot(self, mode_type: str) -> bool:
        """Atomically clear one programming snapshot."""
        _, snap_attr = self._MODE_ATTRS[mode_type]
        with self._chat_lock:
            if getattr(self, snap_attr) is None:
                return False
            setattr(self, snap_attr, None)
            return True

    def clear_programming_snapshot_if_matches(
        self,
        mode_type: str,
        session_id: str,
    ) -> bool:
        """Compare-and-clear a snapshot without erasing a newer replacement."""
        expected = str(session_id or "")
        if not expected:
            return False
        _, snap_attr = self._MODE_ATTRS[mode_type]
        with self._chat_lock:
            snapshot = getattr(self, snap_attr)
            if str(getattr(snapshot, "session_id", "") or "") != expected:
                return False
            setattr(self, snap_attr, None)
            return True

    def commit_acp_programming_activation(
        self,
        mode_type: str,
        tool_name: str,
        model_name: Optional[str],
        session_id: str,
        query_count: int = 0,
        activate_mode: bool = True,
    ) -> dict[str, object]:
        """Atomically select an ACP backend and activate its project mode.

        The returned state is private rollback data for ``ProjectManager`` if
        persistence fails.  Callers must hold the manager lock before invoking
        this method so a persisted activation cannot interleave with another
        project-wide commit.
        """
        if mode_type not in self._MODE_ATTRS:
            raise ValueError(f"unsupported programming mode: {mode_type}")
        with self._chat_lock:
            previous = {
                "acp_tool_name": self.acp_tool_name,
                "acp_model_name": self.acp_model_name,
                "last_active": self.last_active,
                "modes": {
                    name: (getattr(self, flag), getattr(self, snapshot))
                    for name, (flag, snapshot) in self._MODE_ATTRS.items()
                },
            }
            if activate_mode:
                for flag, _snapshot in self._MODE_ATTRS.values():
                    setattr(self, flag, False)
                mode_flag, snapshot_attr = self._MODE_ATTRS[mode_type]
                setattr(self, mode_flag, True)
                setattr(
                    self,
                    snapshot_attr,
                    SessionSnapshot(
                        session_id=session_id,
                        query_count=query_count,
                        last_query="",
                        is_resumable=True,
                    ),
                )
            self.acp_tool_name = tool_name
            self.acp_model_name = model_name
            self.touch()
            return previous

    def replace_acp_configuration(
        self,
        tool_name: str,
        model_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Replace the saved tool/model pair and return rollback state."""
        with self._chat_lock:
            previous = (self.acp_tool_name, self.acp_model_name)
            self.acp_tool_name = tool_name
            self.acp_model_name = model_name
            return previous

    def restore_acp_programming_activation(self, previous: dict[str, object]) -> None:
        """Restore rollback data from a failed persisted ACP activation."""
        with self._chat_lock:
            self.acp_tool_name = previous["acp_tool_name"]  # type: ignore[assignment]
            self.acp_model_name = previous["acp_model_name"]  # type: ignore[assignment]
            self.last_active = float(previous["last_active"])
            for name, (mode_flag, snapshot_attr) in self._MODE_ATTRS.items():
                enabled, snapshot = previous["modes"][name]  # type: ignore[index]
                setattr(self, mode_flag, enabled)
                setattr(self, snapshot_attr, snapshot)

    # ── Backward-compatible delegates ──

    def set_coco_mode(self, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        self.set_programming_mode("coco", enabled, session_id, query_count)

    def update_coco_snapshot(self, query: str, query_count: int):
        self.update_programming_snapshot("coco", query, query_count)

    def set_claude_mode(self, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        self.set_programming_mode("claude", enabled, session_id, query_count)

    def update_claude_snapshot(self, query: str, query_count: int, session_id: Optional[str] = None):
        self.update_programming_snapshot("claude", query, query_count, session_id)

    def set_aiden_mode(self, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        self.set_programming_mode("aiden", enabled, session_id, query_count)

    def update_aiden_snapshot(self, query: str, query_count: int, session_id: Optional[str] = None):
        self.update_programming_snapshot("aiden", query, query_count, session_id)

    def set_codex_mode(self, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        self.set_programming_mode("codex", enabled, session_id, query_count)

    def update_codex_snapshot(self, query: str, query_count: int, session_id: Optional[str] = None):
        self.update_programming_snapshot("codex", query, query_count, session_id)

    def set_gemini_mode(self, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        self.set_programming_mode("gemini", enabled, session_id, query_count)

    def update_gemini_snapshot(self, query: str, query_count: int, session_id: Optional[str] = None):
        self.update_programming_snapshot("gemini", query, query_count, session_id)

    def set_traex_mode(self, enabled: bool, session_id: Optional[str] = None, query_count: int = 0):
        self.set_programming_mode("traex", enabled, session_id, query_count)

    def update_traex_snapshot(self, query: str, query_count: int, session_id: Optional[str] = None):
        self.update_programming_snapshot("traex", query, query_count, session_id)

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
            "conversation_history": [
                {
                    "role": item.role,
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "message_id": item.message_id,
                }
                for item in self.conversation_history
            ],
        }
        with self._chat_lock:
            for mode_type, (mode_flag, snap_attr) in self._MODE_ATTRS.items():
                d[mode_flag] = getattr(self, mode_flag)
                d[snap_attr] = self._snap_to_dict(getattr(self, snap_attr))
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
            coco_mode=data.get("coco_mode", False),
            claude_mode=data.get("claude_mode", False),
            aiden_mode=data.get("aiden_mode", False),
            codex_mode=data.get("codex_mode", False),
            gemini_mode=data.get("gemini_mode", False),
            traex_mode=data.get("traex_mode", False),
            grok_mode=data.get("grok_mode", False),
            dsh_mode=data.get("dsh_mode", False),
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
        if data.get("coco_session_snapshot"):
            snap = data["coco_session_snapshot"]
            ctx.coco_session_snapshot = SessionSnapshot(
                session_id=snap["session_id"],
                query_count=snap["query_count"],
                last_query=snap["last_query"],
                is_resumable=snap.get("is_resumable", True),
            )
        # Restore remaining mode snapshots via generic loop
        for mode_type, (_, snap_attr) in cls._MODE_ATTRS.items():
            if mode_type == "coco":
                continue  # already handled above
            snap_data = data.get(snap_attr)
            if snap_data:
                setattr(ctx, snap_attr, SessionSnapshot(
                    session_id=snap_data["session_id"],
                    query_count=snap_data["query_count"],
                    last_query=snap_data["last_query"],
                    is_resumable=snap_data.get("is_resumable", True),
                ))
        for item_data in data.get("conversation_history", []):
            ctx.conversation_history.append(
                ConversationItem(
                    role=item_data["role"],
                    content=item_data["content"],
                    timestamp=item_data.get("timestamp", time.time()),
                    message_id=item_data.get("message_id"),
                )
            )
        return ctx
