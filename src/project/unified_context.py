"""
项目级统一上下文管理 —— 数据结构与内存存储

解决的核心问题：
- 各编程模式（Coco/Claude/Shell/Deep Engine）上下文彼此隔离
- conversation_history 不持久化（本模块为纯内存，服务重启后重置）
- 模式切换时 AI 丧失前文理解
- 项目切换时上下文处理缺乏安全边界

设计原则：
- 组合而非替换 —— 通过 project_id 与现有 ProjectContext 关联
- 纯内存存储 —— 服务运行期间持续生效，重启后重置
- 统一条目模型 —— 不同类型的上下文数据使用同一 ContextEntry 结构
- 版本快照 —— 在关键节点（模式切换、Deep 完成）打版本书签，支持 diff 查询
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class ContextEntryType(Enum):
    """上下文条目的数据类型"""

    CONVERSATION = "conversation"
    SESSION_SNAPSHOT = "session_snapshot"
    MODE_TRANSITION = "mode_transition"
    DEEP_ENGINE_RESULT = "deep_result"
    AI_SUMMARY = "ai_summary"
    FILE_CHANGE = "file_change"


class ContextSourceMode(Enum):
    """产生该条目的编程模式"""

    SMART = "smart"
    COCO = "coco"
    CLAUDE = "claude"
    AIDEN = "aiden"
    CODEX = "codex"
    GEMINI = "gemini"
    TRAEX = "traex"
    SHELL = "shell"
    DEEP_ENGINE = "deep_engine"


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------


@dataclass
class ContextEntry:
    """
    单条上下文记录。所有类型的上下文数据都通过此结构统一存储。

    字段说明：
        entry_id:     唯一标识，自动生成
        entry_type:   条目类型（对话/快照/模式切换/Deep结果/AI摘要/文件变更）
        source_mode:  产生该条目的编程模式
        content:      主要文本内容
        metadata:     类型特定的结构化附加数据（如 role/message_id/session_id 等）
        created_at:   创建时间戳
    """

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # 单调递增的序号（用于跨滚动窗口的增量 diff）
    seq: int = 0
    entry_type: ContextEntryType = ContextEntryType.CONVERSATION
    source_mode: ContextSourceMode = ContextSourceMode.SMART
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "seq": self.seq,
            "entry_type": self.entry_type.value,
            "source_mode": self.source_mode.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextEntry":
        return cls(
            entry_id=data.get("entry_id", uuid.uuid4().hex[:12]),
            seq=data.get("seq", 0),
            entry_type=ContextEntryType(data.get("entry_type", "conversation")),
            source_mode=ContextSourceMode(data.get("source_mode", "smart")),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ContextVersion:
    """
    轻量级版本书签 —— 标记上下文在某个关键时刻的状态。

    版本不存储完整快照，而是记录当时的 entry_count，
    配合 UnifiedContext.get_entries_since_version() 实现增量 diff。

    字段说明：
        version_id:      唯一标识
        version_number:  单调递增的版本号
        reason:          创建原因（如 "mode_transition: coco -> claude"）
        source_mode:     触发版本创建的模式
        summary:         该版本时刻的上下文摘要文本
        entry_count:     版本创建时 entries 列表的长度（用于 diff 计算）
        created_at:      创建时间戳
    """

    version_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    version_number: int = 0
    reason: str = ""
    source_mode: ContextSourceMode = ContextSourceMode.SMART
    summary: str = ""
    entry_count: int = 0
    # 版本创建时刻的最后一个 entry.seq（用于在滚动窗口淘汰后仍能计算增量 diff）
    last_seq: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "reason": self.reason,
            "source_mode": self.source_mode.value,
            "summary": self.summary,
            "entry_count": self.entry_count,
            "last_seq": self.last_seq,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextVersion":
        return cls(
            version_id=data.get("version_id", uuid.uuid4().hex[:8]),
            version_number=data.get("version_number", 0),
            reason=data.get("reason", ""),
            source_mode=ContextSourceMode(data.get("source_mode", "smart")),
            summary=data.get("summary", ""),
            entry_count=data.get("entry_count", 0),
            last_seq=data.get("last_seq", 0),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ContextBridgeSummary:
    """
    跨模式上下文桥接摘要。

    当用户从一个编程模式切换到另一个时（如 Coco -> Claude），
    系统从近期上下文条目中生成此摘要，然后通过 to_injection_prompt()
    格式化为文本，注入到新模式 session 的首条 prompt 中。

    这是唯一可行的跨模式上下文传递方式，因为 AI 对话状态
    存储在 CLI 子进程的服务端，GhostAP 只能通过 prompt 文本传递信息。
    """

    from_mode: ContextSourceMode = ContextSourceMode.SMART
    to_mode: ContextSourceMode = ContextSourceMode.SMART
    summary_text: str = ""
    key_decisions: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "summary_text": self.summary_text,
            "key_decisions": self.key_decisions,
            "files_modified": self.files_modified,
            "pending_tasks": self.pending_tasks,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextBridgeSummary":
        return cls(
            from_mode=ContextSourceMode(data.get("from_mode", "smart")),
            to_mode=ContextSourceMode(data.get("to_mode", "smart")),
            summary_text=data.get("summary_text", ""),
            key_decisions=data.get("key_decisions", []),
            files_modified=data.get("files_modified", []),
            pending_tasks=data.get("pending_tasks", []),
            created_at=data.get("created_at", time.time()),
        )

    def to_injection_prompt(self) -> str:
        """格式化为可注入到新 session 首条 prompt 的文本"""
        parts = [f"[Context from previous {self.from_mode.value} session]"]
        if self.summary_text:
            parts.append(self.summary_text)
        if self.key_decisions:
            parts.append("Key decisions:")
            parts.extend(f"  - {d}" for d in self.key_decisions)
        if self.files_modified:
            parts.append(f"Files modified: {', '.join(self.files_modified)}")
        if self.pending_tasks:
            parts.append("Pending tasks:")
            parts.extend(f"  - {t}" for t in self.pending_tasks)
        parts.append("[End of context]")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 顶层上下文容器
# ---------------------------------------------------------------------------


class UnifiedContext:
    """
    单个项目的统一上下文容器（纯内存）。

    通过 project_id 与现有 ProjectContext 一一对应。
    管理上下文条目列表、版本书签和跨模式桥接摘要。

    容量限制：
        - entries:  滚动窗口，默认保留最近 200 条
        - versions: 默认保留最近 50 个版本

    Thread-safety:
        All mutable state (_entries, _versions, _last_bridge, _entry_index,
        _next_seq, _current_version_number) is protected by ``_mu``.
        Callers may invoke any public method from any thread.
    """

    def __init__(self, project_id: str, max_entries: int = 200, max_versions: int = 50):
        self.project_id: str = project_id
        self.max_entries: int = max_entries
        self.max_versions: int = max_versions
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

        # Internal lock protecting all mutable state below.
        self._mu = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        self._entries: list[ContextEntry] = []
        self._versions: list[ContextVersion] = []
        self._current_version_number: int = 0
        self._last_bridge: Optional[ContextBridgeSummary] = None

        # 单调递增 entry 序号（不因滚动窗口淘汰而回退）
        self._next_seq: int = 1

        # entry_id -> list index 的快速查找映射
        self._entry_index: dict[str, int] = {}

    # ---- 属性 ----

    @property
    def entries(self) -> list[ContextEntry]:
        with self._mu:
            return list(self._entries)

    @property
    def versions(self) -> list[ContextVersion]:
        with self._mu:
            return list(self._versions)

    @property
    def current_version_number(self) -> int:
        with self._mu:
            return self._current_version_number

    @property
    def last_bridge_summary(self) -> Optional[ContextBridgeSummary]:
        with self._mu:
            return self._last_bridge

    @property
    def entry_count(self) -> int:
        with self._mu:
            return len(self._entries)

    # ---- Create: 添加条目 ----

    def add_entry(self, entry: ContextEntry) -> ContextEntry:
        """添加一条上下文记录，超出容量时淘汰最旧的条目"""
        with self._mu:
            # 为新 entry 分配单调递增的 seq，便于版本 diff
            try:
                if getattr(entry, "seq", 0) <= 0:
                    entry.seq = self._next_seq
                    self._next_seq += 1
            except Exception:
                # 极端情况下（比如被外部替换为非 dataclass 对象）兜底不影响主流程
                logger.debug("[Context] seq assignment failed", exc_info=True)

            self._entries.append(entry)
            self._entry_index[entry.entry_id] = len(self._entries) - 1

            if self.max_entries > 0 and len(self._entries) > self.max_entries:
                evicted = len(self._entries) - self.max_entries
                self._entries = self._entries[-self.max_entries :]
                self._rebuild_index()
                logger.debug("[Context:%s] 淘汰 %d 条旧条目，当前 %d 条", self.project_id, evicted, len(self._entries))

            self.updated_at = time.time()
            return entry

    def add_conversation(
        self,
        role: str,
        content: str,
        source_mode: ContextSourceMode,
        message_id: Optional[str] = None,
    ) -> ContextEntry:
        """便捷方法：添加对话类型条目"""
        entry = ContextEntry(
            entry_type=ContextEntryType.CONVERSATION,
            source_mode=source_mode,
            content=content,
            metadata={"role": role, "message_id": message_id},
        )
        return self.add_entry(entry)

    def add_session_snapshot(
        self,
        session_data: dict,
        source_mode: ContextSourceMode,
    ) -> ContextEntry:
        """便捷方法：添加会话快照条目（模式退出时调用）"""
        entry = ContextEntry(
            entry_type=ContextEntryType.SESSION_SNAPSHOT,
            source_mode=source_mode,
            content=f"Session ended: {session_data.get('session_id', 'unknown')}",
            metadata=session_data,
        )
        return self.add_entry(entry)

    def add_mode_transition(
        self,
        from_mode: ContextSourceMode,
        to_mode: ContextSourceMode,
        reason: str = "",
    ) -> ContextEntry:
        """便捷方法：记录模式切换事件"""
        entry = ContextEntry(
            entry_type=ContextEntryType.MODE_TRANSITION,
            source_mode=from_mode,
            content=f"{from_mode.value} -> {to_mode.value}",
            metadata={
                "from_mode": from_mode.value,
                "to_mode": to_mode.value,
                "reason": reason,
            },
        )
        return self.add_entry(entry)

    def add_deep_engine_result(self, deep_project_data: dict) -> ContextEntry:
        """便捷方法：记录 Deep Engine 任务结果"""
        entry = ContextEntry(
            entry_type=ContextEntryType.DEEP_ENGINE_RESULT,
            source_mode=ContextSourceMode.DEEP_ENGINE,
            content=f"Deep Engine completed: {deep_project_data.get('name', 'unknown')}",
            metadata=deep_project_data,
        )
        return self.add_entry(entry)

    # ---- Read: 查询 ----

    def get_entry(self, entry_id: str) -> Optional[ContextEntry]:
        """按 entry_id 精确查找，O(1)"""
        with self._mu:
            idx = self._entry_index.get(entry_id)
            if idx is not None and idx < len(self._entries):
                entry = self._entries[idx]
                if entry.entry_id == entry_id:
                    return entry
            # 索引失效时回退线性查找
            for entry in self._entries:
                if entry.entry_id == entry_id:
                    return entry
            return None

    def get_entries_by_type(self, entry_type: ContextEntryType) -> list[ContextEntry]:
        """按条目类型筛选"""
        with self._mu:
            return [e for e in self._entries if e.entry_type == entry_type]

    def get_entries_by_mode(self, source_mode: ContextSourceMode) -> list[ContextEntry]:
        """按来源模式筛选"""
        with self._mu:
            return [e for e in self._entries if e.source_mode == source_mode]

    def get_recent_entries(self, limit: int = 10) -> list[ContextEntry]:
        """获取最近 N 条条目"""
        with self._mu:
            return list(self._entries[-limit:])

    def get_conversations(self, limit: int = 20) -> list[ContextEntry]:
        """获取最近 N 条对话记录"""
        with self._mu:
            convs = [e for e in self._entries if e.entry_type == ContextEntryType.CONVERSATION]
            return convs[-limit:]

    def query_entries(
        self,
        entry_type: Optional[ContextEntryType] = None,
        source_mode: Optional[ContextSourceMode] = None,
        since: Optional[float] = None,
        limit: int = 50,
    ) -> list[ContextEntry]:
        """组合条件查询"""
        with self._mu:
            results = self._entries
            if entry_type is not None:
                results = [e for e in results if e.entry_type == entry_type]
            if source_mode is not None:
                results = [e for e in results if e.source_mode == source_mode]
            if since is not None:
                results = [e for e in results if e.created_at >= since]
            return results[-limit:]

    # ---- Update: 更新条目 ----

    def update_entry(self, entry_id: str, content: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        """按 entry_id 更新条目的 content 或 metadata"""
        with self._mu:
            entry = self._get_entry_unlocked(entry_id)
            if entry is None:
                return False
            if content is not None:
                entry.content = content
            if metadata is not None:
                entry.metadata.update(metadata)
            self.updated_at = time.time()
            return True

    # ---- Delete: 删除条目 ----

    def remove_entry(self, entry_id: str) -> bool:
        """按 entry_id 删除条目"""
        with self._mu:
            for i, entry in enumerate(self._entries):
                if entry.entry_id == entry_id:
                    self._entries.pop(i)
                    self._rebuild_index()
                    self.updated_at = time.time()
                    return True
            return False

    def clear_entries(self) -> int:
        """清空所有条目，返回被清除的数量"""
        with self._mu:
            count = len(self._entries)
            self._entries.clear()
            self._entry_index.clear()
            self.updated_at = time.time()
            return count

    def clear_entries_by_mode(self, source_mode: ContextSourceMode) -> int:
        """清除指定模式的所有条目"""
        with self._mu:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.source_mode != source_mode]
            self._rebuild_index()
            removed = before - len(self._entries)
            if removed > 0:
                self.updated_at = time.time()
            return removed

    # ---- 版本控制 ----

    def create_version(
        self,
        reason: str,
        source_mode: ContextSourceMode,
        summary: str = "",
    ) -> ContextVersion:
        """在当前时刻创建版本书签"""
        with self._mu:
            self._current_version_number += 1
            last_seq = 0
            if self._entries:
                try:
                    last_seq = getattr(self._entries[-1], "seq", 0) or 0
                except Exception:
                    last_seq = 0
            version = ContextVersion(
                version_number=self._current_version_number,
                reason=reason,
                source_mode=source_mode,
                summary=summary,
                entry_count=len(self._entries),
                last_seq=last_seq,
            )
            self._versions.append(version)
            if len(self._versions) > self.max_versions:
                self._versions = self._versions[-self.max_versions :]
            self.updated_at = time.time()
            logger.debug("[Context:%s] 创建版本 v%d: %s", self.project_id, version.version_number, reason)
            return version

    def get_version(self, version_number: int) -> Optional[ContextVersion]:
        """按版本号查找"""
        with self._mu:
            for v in self._versions:
                if v.version_number == version_number:
                    return v
            return None

    def get_entries_since_version(self, version_number: int) -> list[ContextEntry]:
        """获取某个版本之后新增的所有条目（增量 diff）"""
        with self._mu:
            version = None
            for v in self._versions:
                if v.version_number == version_number:
                    version = v
                    break
            if version is None:
                return list(self._entries)

            # 优先使用 seq 做增量 diff
            last_seq = getattr(version, "last_seq", 0) or 0
            if last_seq > 0:
                results: list[ContextEntry] = []
                for e in self._entries:
                    try:
                        if getattr(e, "seq", 0) > last_seq:
                            results.append(e)
                    except Exception:
                        continue
                return results

            if len(self._entries) <= version.entry_count:
                return []
            return list(self._entries[version.entry_count :])

    # ---- 跨模式桥接 ----

    def build_bridge_summary(
        self,
        from_mode: ContextSourceMode,
        to_mode: ContextSourceMode,
        max_items: int = 10,
    ) -> ContextBridgeSummary:
        """
        从近期条目构建跨模式桥接摘要。
        """
        with self._mu:
            bridgeable_types = {
                ContextEntryType.CONVERSATION,
                ContextEntryType.AI_SUMMARY,
                ContextEntryType.DEEP_ENGINE_RESULT,
                ContextEntryType.FILE_CHANGE,
            }
            recent: list[ContextEntry] = []
            for entry in reversed(self._entries):
                if len(recent) >= max_items:
                    break
                if entry.entry_type in bridgeable_types:
                    recent.append(entry)
            recent.reverse()

            conversation_lines: list[str] = []
            files_modified: list[str] = []
            for entry in recent:
                if entry.entry_type == ContextEntryType.CONVERSATION:
                    role = entry.metadata.get("role", "unknown")
                    text = entry.content[:300]
                    conversation_lines.append(f"{role}: {text}")
                elif entry.entry_type == ContextEntryType.DEEP_ENGINE_RESULT:
                    tasks = entry.metadata.get("tasks", [])
                    for t in tasks:
                        if t.get("status") == "completed" and t.get("result"):
                            conversation_lines.append(f"[completed task] {t.get('title', '')}: {t['result'][:150]}")
                elif entry.entry_type == ContextEntryType.FILE_CHANGE:
                    files_modified.append(entry.content)

            bridge = ContextBridgeSummary(
                from_mode=from_mode,
                to_mode=to_mode,
                summary_text="\n".join(conversation_lines[-8:]),
                files_modified=files_modified,
            )
            self._last_bridge = bridge
            self.updated_at = time.time()
            logger.info(
                "[Context:%s] 构建桥接摘要: %s -> %s, %d 条对话, %d 个文件变更",
                self.project_id,
                from_mode.value,
                to_mode.value,
                len(conversation_lines),
                len(files_modified),
            )
            return bridge

    def consume_bridge_summary(self) -> Optional[ContextBridgeSummary]:
        """取出并消费桥接摘要（取出后清空，避免重复注入）"""
        with self._mu:
            bridge = self._last_bridge
            self._last_bridge = None
            if bridge:
                logger.info(
                    "[Context:%s] 消费桥接摘要: %s -> %s", self.project_id, bridge.from_mode.value, bridge.to_mode.value
                )
            return bridge

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        with self._mu:
            return {
                "project_id": self.project_id,
                "entries": [e.to_dict() for e in self._entries],
                "versions": [v.to_dict() for v in self._versions],
                "current_version_number": self._current_version_number,
                "last_bridge_summary": self._last_bridge.to_dict() if self._last_bridge else None,
                "max_entries": self.max_entries,
                "max_versions": self.max_versions,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedContext":
        ctx = cls(
            project_id=data["project_id"],
            max_entries=data.get("max_entries", 200),
            max_versions=data.get("max_versions", 50),
        )
        ctx.created_at = data.get("created_at", time.time())
        ctx.updated_at = data.get("updated_at", time.time())
        ctx._current_version_number = data.get("current_version_number", 0)
        ctx._entries = [ContextEntry.from_dict(e) for e in data.get("entries", [])]
        ctx._versions = [ContextVersion.from_dict(v) for v in data.get("versions", [])]
        if data.get("last_bridge_summary"):
            ctx._last_bridge = ContextBridgeSummary.from_dict(data["last_bridge_summary"])
        # 恢复 seq 自增计数器
        try:
            max_seq = 0
            for e in ctx._entries:
                max_seq = max(max_seq, getattr(e, "seq", 0) or 0)
            ctx._next_seq = max_seq + 1 if max_seq > 0 else 1
        except Exception:
            ctx._next_seq = 1
        ctx._rebuild_index()
        return ctx

    # ---- 内部方法 ----

    def _get_entry_unlocked(self, entry_id: str) -> Optional[ContextEntry]:
        """按 entry_id 查找（调用方必须已持有 _mu）。"""
        idx = self._entry_index.get(entry_id)
        if idx is not None and idx < len(self._entries):
            entry = self._entries[idx]
            if entry.entry_id == entry_id:
                return entry
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def _rebuild_index(self):
        """重建 entry_id -> index 映射（调用方必须已持有 _mu）"""
        self._entry_index = {e.entry_id: i for i, e in enumerate(self._entries)}


# ---------------------------------------------------------------------------
# 内存存储管理器
# ---------------------------------------------------------------------------


class UnifiedContextStore:
    """
    项目级上下文的内存存储管理器。

    特性：
        - 以 {chat_id}:{project_id} 复合键存储，每个群×项目独立的 UnifiedContext 实例
        - 不同群绑定同一项目时对话历史、bridge summary、版本书签完全隔离
        - 服务运行期间持续生效，服务重启后自动重置
        - 线程安全（threading.Lock）
        - O(1) 的项目上下文查找
        - 支持全局统计与批量清理
    """

    def __init__(self, default_max_entries: int = 200, default_max_versions: int = 50):
        self._store: dict[str, UnifiedContext] = {}
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._default_max_entries = default_max_entries
        self._default_max_versions = default_max_versions

    @staticmethod
    def _composite_key(chat_id: str, project_id: str) -> str:
        """Build the ``{chat_id}:{project_id}`` composite key."""
        return f"{chat_id}:{project_id}"

    # ---- Create / Read ----

    def get_or_create(self, project_id: str, *, chat_id: str = "") -> UnifiedContext:
        """获取项目的上下文，不存在则自动创建"""
        key = self._composite_key(chat_id, project_id)
        with self._lock:
            if key not in self._store:
                self._store[key] = UnifiedContext(
                    project_id=project_id,
                    max_entries=self._default_max_entries,
                    max_versions=self._default_max_versions,
                )
                logger.info("[ContextStore] 为 %s 创建统一上下文", key)
            return self._store[key]

    def get(self, project_id: str, *, chat_id: str = "") -> Optional[UnifiedContext]:
        """获取项目的上下文，不存在返回 None"""
        key = self._composite_key(chat_id, project_id)
        with self._lock:
            return self._store.get(key)

    def has(self, project_id: str, *, chat_id: str = "") -> bool:
        """检查项目上下文是否存在"""
        key = self._composite_key(chat_id, project_id)
        with self._lock:
            return key in self._store

    def list_project_ids(self) -> list[str]:
        """列出所有有上下文的复合键"""
        with self._lock:
            return list(self._store.keys())

    # ---- Delete ----

    def remove(self, project_id: str, *, chat_id: str = "") -> bool:
        """移除项目的上下文，返回是否成功"""
        key = self._composite_key(chat_id, project_id)
        with self._lock:
            removed = self._store.pop(key, None) is not None
            if removed:
                logger.info("[ContextStore] 移除 %s 的统一上下文", key)
            return removed

    def clear(self) -> int:
        """清空所有项目的上下文，返回被清除的项目数"""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    # ---- 统计 ----

    def stats(self) -> dict[str, Any]:
        """返回全局统计信息"""
        with self._lock:
            total_entries = sum(ctx.entry_count for ctx in self._store.values())
            total_versions = sum(len(ctx.versions) for ctx in self._store.values())
            return {
                "project_count": len(self._store),
                "total_entries": total_entries,
                "total_versions": total_versions,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# 标准化响应
# ---------------------------------------------------------------------------


@dataclass
class ContextResult:
    """
    所有上下文操作的标准化返回值。

    调用方只需检查 success 即可判断操作是否成功，
    无需 try/except 或空值判断。

    字段说明：
        success:    操作是否成功
        message:    人类可读的结果描述（成功或失败原因）
        data:       操作返回的数据（类型取决于具体操作）
        project_id: 操作涉及的项目 ID
    """

    success: bool
    message: str
    data: Optional[Any] = None
    project_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 项目级上下文管理接口
# ---------------------------------------------------------------------------


class ProjectContextManager:
    """
    项目级上下文的核心操作接口。

    在底层 UnifiedContextStore 之上提供五个标准操作：
        - create_context:   为指定项目创建上下文
        - get_context:      查询指定项目的上下文
        - update_context:   向指定项目的上下文追加/修改数据
        - delete_context:   删除指定项目的上下文
        - context_exists:   检查指定项目的上下文是否存在

    所有操作：
        - 接受 project_id 作为项目标识参数
        - 返回统一的 ContextResult 响应
        - 线程安全（继承自 UnifiedContextStore 的锁机制）
        - 对非法参数做防御性校验
    """

    def __init__(self, store: Optional[UnifiedContextStore] = None):
        self._store = store or UnifiedContextStore()

    @property
    def store(self) -> UnifiedContextStore:
        return self._store

    # ---- 1. createContext ----

    def create_context(
        self,
        project_id: str,
        initial_entries: Optional[list[ContextEntry]] = None,
        max_entries: int = 200,
        max_versions: int = 50,
        *,
        chat_id: str = "",
    ) -> ContextResult:
        """
        为指定项目创建上下文。

        如果该项目已有上下文，返回失败（不会覆盖）。
        可选传入初始条目列表和容量配置。

        Args:
            project_id:      项目唯一标识
            initial_entries: 可选的初始条目列表
            max_entries:     条目滚动窗口上限
            max_versions:    版本数上限
            chat_id:         聊天会话标识，用于多会话隔离

        Returns:
            ContextResult，data 字段为创建的 UnifiedContext
        """
        if not project_id or not project_id.strip():
            return ContextResult(
                success=False,
                message="project_id 不能为空",
                project_id=project_id,
            )

        if self._store.has(project_id, chat_id=chat_id):
            return ContextResult(
                success=False,
                message=f"项目 {project_id} 的上下文已存在",
                data=self._store.get(project_id, chat_id=chat_id),
                project_id=project_id,
            )

        ctx = UnifiedContext(
            project_id=project_id,
            max_entries=max_entries,
            max_versions=max_versions,
        )

        if initial_entries:
            for entry in initial_entries:
                ctx.add_entry(entry)

        # 直接写入 store 内部（需要穿过锁）
        key = self._store._composite_key(chat_id, project_id)
        with self._store._lock:
            self._store._store[key] = ctx

        return ContextResult(
            success=True,
            message=f"项目 {project_id} 的上下文已创建",
            data=ctx,
            project_id=project_id,
        )

    # ---- 2. getContext ----

    def get_context(
        self,
        project_id: str,
        include_entries: bool = True,
        entry_type: Optional[ContextEntryType] = None,
        source_mode: Optional[ContextSourceMode] = None,
        recent_limit: Optional[int] = None,
        *,
        chat_id: str = "",
    ) -> ContextResult:
        """
        查询指定项目的上下文。

        支持多种查询模式：
        - 获取完整上下文对象
        - 按条目类型筛选
        - 按来源模式筛选
        - 只获取最近 N 条

        Args:
            project_id:      项目唯一标识
            include_entries:  是否在 data 中包含条目列表（False 时只返回元信息）
            entry_type:      按条目类型筛选
            source_mode:     按来源模式筛选
            recent_limit:    只返回最近 N 条条目
            chat_id:         聊天会话标识，用于多会话隔离

        Returns:
            ContextResult，data 字段为查询结果 dict
        """
        if not project_id or not project_id.strip():
            return ContextResult(
                success=False,
                message="project_id 不能为空",
                project_id=project_id,
            )

        ctx = self._store.get(project_id, chat_id=chat_id)
        if ctx is None:
            return ContextResult(
                success=False,
                message=f"项目 {project_id} 的上下文不存在",
                project_id=project_id,
            )

        result_data: dict[str, Any] = {
            "project_id": ctx.project_id,
            "entry_count": ctx.entry_count,
            "version_count": len(ctx.versions),
            "current_version": ctx.current_version_number,
            "created_at": ctx.created_at,
            "updated_at": ctx.updated_at,
            "has_bridge_summary": ctx.last_bridge_summary is not None,
        }

        if include_entries:
            if entry_type is not None or source_mode is not None or recent_limit is not None:
                entries = ctx.query_entries(
                    entry_type=entry_type,
                    source_mode=source_mode,
                    limit=recent_limit or 50,
                )
            else:
                entries = ctx.entries
            result_data["entries"] = entries
        else:
            result_data["entries"] = []

        return ContextResult(
            success=True,
            message=f"查询成功，共 {ctx.entry_count} 条记录",
            data=result_data,
            project_id=project_id,
        )

    # ---- 3. updateContext ----

    def update_context(
        self,
        project_id: str,
        entries: Optional[list[ContextEntry]] = None,
        conversation: Optional[dict] = None,
        session_snapshot: Optional[dict] = None,
        mode_transition: Optional[dict] = None,
        deep_result: Optional[dict] = None,
        create_if_missing: bool = True,
        *,
        chat_id: str = "",
    ) -> ContextResult:
        """
        更新指定项目的上下文：追加条目或批量写入。

        支持两种写入方式：
        1. 传入 entries 列表直接批量追加
        2. 使用便捷参数（conversation/session_snapshot/mode_transition/deep_result）

        Args:
            project_id:        项目唯一标识
            entries:           要追加的条目列表
            conversation:      便捷参数 {"role": str, "content": str, "source_mode": str,
                               "message_id": Optional[str]}
            session_snapshot:  便捷参数 {"data": dict, "source_mode": str}
            mode_transition:   便捷参数 {"from_mode": str, "to_mode": str,
                               "reason": Optional[str]}
            deep_result:       便捷参数 {"data": dict} (Deep Engine 结果)
            create_if_missing: 上下文不存在时是否自动创建

        Returns:
            ContextResult，data 字段为 {"added_count": int, "total_count": int}
        """
        if not project_id or not project_id.strip():
            return ContextResult(
                success=False,
                message="project_id 不能为空",
                project_id=project_id,
            )

        ctx = self._store.get(project_id, chat_id=chat_id)
        if ctx is None:
            if not create_if_missing:
                return ContextResult(
                    success=False,
                    message=f"项目 {project_id} 的上下文不存在",
                    project_id=project_id,
                )
            ctx = self._store.get_or_create(project_id, chat_id=chat_id)

        added = 0

        # 方式 1: 直接追加条目列表
        if entries:
            for entry in entries:
                ctx.add_entry(entry)
                added += 1

        # 方式 2: 便捷参数
        if conversation:
            source = ContextSourceMode(conversation.get("source_mode", "smart"))
            ctx.add_conversation(
                role=conversation["role"],
                content=conversation["content"],
                source_mode=source,
                message_id=conversation.get("message_id"),
            )
            added += 1

        if session_snapshot:
            source = ContextSourceMode(session_snapshot.get("source_mode", "smart"))
            ctx.add_session_snapshot(
                session_data=session_snapshot["data"],
                source_mode=source,
            )
            added += 1

        if mode_transition:
            from_m = ContextSourceMode(mode_transition["from_mode"])
            to_m = ContextSourceMode(mode_transition["to_mode"])
            ctx.add_mode_transition(
                from_mode=from_m,
                to_mode=to_m,
                reason=mode_transition.get("reason", ""),
            )
            added += 1

        if deep_result:
            ctx.add_deep_engine_result(deep_result["data"])
            added += 1

        if added == 0:
            return ContextResult(
                success=False,
                message="未提供任何要更新的数据",
                data={"added_count": 0, "total_count": ctx.entry_count},
                project_id=project_id,
            )

        return ContextResult(
            success=True,
            message=f"已追加 {added} 条记录",
            data={"added_count": added, "total_count": ctx.entry_count},
            project_id=project_id,
        )

    # ---- 4. deleteContext ----

    def delete_context(
        self,
        project_id: str,
        entry_id: Optional[str] = None,
        source_mode: Optional[ContextSourceMode] = None,
        *,
        chat_id: str = "",
    ) -> ContextResult:
        """
        删除上下文数据。

        三种粒度：
        - 不传可选参数：删除整个项目的上下文
        - 传 entry_id：删除单条条目
        - 传 source_mode：删除该模式产生的所有条目

        Args:
            project_id:   项目唯一标识
            entry_id:     要删除的单条条目 ID
            source_mode:  要清除的模式（删除该模式的所有条目）

        Returns:
            ContextResult，data 字段为 {"removed_count": int}
        """
        if not project_id or not project_id.strip():
            return ContextResult(
                success=False,
                message="project_id 不能为空",
                project_id=project_id,
            )

        # 删除单条条目
        if entry_id is not None:
            ctx = self._store.get(project_id, chat_id=chat_id)
            if ctx is None:
                return ContextResult(
                    success=False,
                    message=f"项目 {project_id} 的上下文不存在",
                    project_id=project_id,
                )
            if ctx.remove_entry(entry_id):
                return ContextResult(
                    success=True,
                    message=f"条目 {entry_id} 已删除",
                    data={"removed_count": 1},
                    project_id=project_id,
                )
            return ContextResult(
                success=False,
                message=f"条目 {entry_id} 不存在",
                data={"removed_count": 0},
                project_id=project_id,
            )

        # 删除指定模式的所有条目
        if source_mode is not None:
            ctx = self._store.get(project_id, chat_id=chat_id)
            if ctx is None:
                return ContextResult(
                    success=False,
                    message=f"项目 {project_id} 的上下文不存在",
                    project_id=project_id,
                )
            removed = ctx.clear_entries_by_mode(source_mode)
            return ContextResult(
                success=True,
                message=f"已清除 {source_mode.value} 模式的 {removed} 条记录",
                data={"removed_count": removed},
                project_id=project_id,
            )

        # 删除整个项目上下文
        if self._store.remove(project_id, chat_id=chat_id):
            return ContextResult(
                success=True,
                message=f"项目 {project_id} 的上下文已删除",
                data={"removed_count": 1},
                project_id=project_id,
            )
        return ContextResult(
            success=False,
            message=f"项目 {project_id} 的上下文不存在",
            data={"removed_count": 0},
            project_id=project_id,
        )

    # ---- 5. contextExists ----

    def context_exists(self, project_id: str, *, chat_id: str = "") -> ContextResult:
        """
        检查指定项目的上下文是否存在。

        Args:
            project_id: 项目唯一标识
            chat_id:    聊天会话标识，用于多会话隔离

        Returns:
            ContextResult，data 字段为 {"exists": bool, "entry_count": int}
        """
        if not project_id or not project_id.strip():
            return ContextResult(
                success=True,
                message="project_id 为空",
                data={"exists": False, "entry_count": 0},
                project_id=project_id,
            )

        ctx = self._store.get(project_id, chat_id=chat_id)
        exists = ctx is not None
        entry_count = ctx.entry_count if ctx else 0

        return ContextResult(
            success=True,
            message=f"项目 {project_id} 上下文{'存在' if exists else '不存在'}",
            data={"exists": exists, "entry_count": entry_count},
            project_id=project_id,
        )
