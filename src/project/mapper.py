import threading
import time
from collections import OrderedDict
from typing import Optional


class MessageProjectMapper:
    def __init__(self, ttl: int = 86400, max_size: int = 10000):
        self._map: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._ttl = ttl
        self._max_size = max_size
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def register(self, message_id: str, project_id: str):
        with self._lock:
            current_time = time.time()
            self._map[message_id] = (project_id, current_time)
            self._cleanup()

    def get_project_id(self, message_id: str) -> Optional[str]:
        with self._lock:
            if message_id not in self._map:
                return None

            project_id, timestamp = self._map[message_id]
            if time.time() - timestamp > self._ttl:
                del self._map[message_id]
                return None

            return project_id

    def _cleanup(self):
        current_time = time.time()

        expired_keys = []
        for msg_id, (_, timestamp) in self._map.items():
            if current_time - timestamp > self._ttl:
                expired_keys.append(msg_id)
            else:
                break

        for key in expired_keys:
            del self._map[key]

        while len(self._map) > self._max_size:
            self._map.popitem(last=False)

    def clear(self):
        with self._lock:
            self._map.clear()

    def __len__(self) -> int:
        return len(self._map)


class MessageLinker:
    """在内存中维护“原始消息 ↔ 回复消息 ↔ 任务 run_id ↔ request_id”的关联。

    - 主要用于：
      1) 在回复内容里展示原始消息标识符（便于排查与定位）
      2) 在任务/进展卡片与用户触发消息之间建立可查询的关联
    - 纯内存 + TTL 清理：服务重启后会丢失（符合当前项目的内存态设计）。
    """

    def __init__(self, ttl: int = 86400, max_size: int = 20000):
        self._ttl = ttl
        self._max_size = max_size
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # origin_message_id -> (record, ts)
        self._origins: OrderedDict[str, tuple[dict, float]] = OrderedDict()

        # reverse indexes
        self._reply_to_origin: dict[str, str] = {}
        self._run_to_origin: dict[str, str] = {}
        self._request_to_origin: dict[str, str] = {}

    def register_origin(
        self,
        origin_message_id: str,
        *,
        request_id: str | None = None,
        chat_id: str | None = None,
        project_id: str | None = None,
        chat_type: str | None = None,
        sender_id: str | None = None,
        tenant_key: str | None = None,
    ) -> None:
        if not origin_message_id:
            return
        with self._lock:
            now = time.time()
            rec = self._origins.get(origin_message_id, (None, now))[0] or {
                "origin_message_id": origin_message_id,
                "request_id": None,
                "chat_id": None,
                "project_id": None,
                "chat_type": None,
                "sender_id": None,
                "tenant_key": None,
                "reply_message_ids": [],
                "task_run_ids": [],
            }
            if request_id:
                rec["request_id"] = request_id
                self._request_to_origin[request_id] = origin_message_id
            if chat_id:
                rec["chat_id"] = chat_id
            if project_id:
                rec["project_id"] = project_id
            if chat_type:
                rec["chat_type"] = chat_type
            if sender_id:
                rec["sender_id"] = sender_id
            if tenant_key:
                rec["tenant_key"] = tenant_key
            self._origins[origin_message_id] = (rec, now)
            self._cleanup_unlocked(now)

    def get_request_id(self, origin_message_id: str) -> Optional[str]:
        with self._lock:
            item = self._origins.get(origin_message_id)
            if not item:
                return None
            rec, ts = item
            if time.time() - ts > self._ttl:
                # expire
                self._evict_origin_unlocked(origin_message_id)
                return None
            return rec.get("request_id")

    def register_trusted_origin_if_absent(
        self,
        origin_message_id: str,
        *,
        chat_id: str,
        sender_id: str,
        chat_type: str,
    ) -> bool:
        """Atomically create a complete card provenance record.

        This is intentionally compare-and-set: a partial or conflicting record
        is never upgraded or overwritten by a Chat API fallback.
        """
        if not all(
            isinstance(value, str) and bool(value)
            for value in (origin_message_id, chat_id, sender_id, chat_type)
        ):
            return False
        if chat_type not in {"p2p", "group", "topic_group"}:
            return False
        with self._lock:
            now = time.time()
            item = self._origins.get(origin_message_id)
            if item is not None:
                rec, timestamp = item
                if now - timestamp > self._ttl:
                    self._evict_origin_unlocked(origin_message_id)
                else:
                    return (
                        rec.get("origin_message_id") == origin_message_id
                        and rec.get("chat_id") == chat_id
                        and rec.get("sender_id") == sender_id
                        and rec.get("chat_type") == chat_type
                    )
            self._origins[origin_message_id] = (
                {
                    "origin_message_id": origin_message_id,
                    "request_id": None,
                    "chat_id": chat_id,
                    "project_id": None,
                    "chat_type": chat_type,
                    "sender_id": sender_id,
                    "tenant_key": None,
                    "reply_message_ids": [],
                    "task_run_ids": [],
                },
                now,
            )
            self._cleanup_unlocked(now)
            return True

    def register_trusted_origin_with_tenant(
        self,
        origin_message_id: str,
        *,
        chat_id: str,
        sender_id: str,
        chat_type: str,
        tenant_key: str,
    ) -> bool:
        """Atomically validate trusted coordinates and attach their tenant."""

        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                origin_message_id,
                chat_id,
                sender_id,
                chat_type,
                tenant_key,
            )
        ):
            return False
        if chat_type not in {"p2p", "group", "topic_group"}:
            return False
        with self._lock:
            now = time.time()
            item = self._origins.get(origin_message_id)
            if item is not None:
                rec, timestamp = item
                if now - timestamp > self._ttl:
                    self._evict_origin_unlocked(origin_message_id)
                else:
                    if (
                        rec.get("origin_message_id") != origin_message_id
                        or rec.get("chat_id") != chat_id
                        or rec.get("sender_id") != sender_id
                        or rec.get("chat_type") != chat_type
                        or rec.get("tenant_key") not in {None, "", tenant_key}
                    ):
                        return False
                    rec["tenant_key"] = tenant_key
                    self._origins[origin_message_id] = (rec, now)
                    self._cleanup_unlocked(now)
                    return True
            self._origins[origin_message_id] = (
                {
                    "origin_message_id": origin_message_id,
                    "request_id": None,
                    "chat_id": chat_id,
                    "project_id": None,
                    "chat_type": chat_type,
                    "sender_id": sender_id,
                    "tenant_key": tenant_key,
                    "reply_message_ids": [],
                    "task_run_ids": [],
                },
                now,
            )
            self._cleanup_unlocked(now)
            return True

    def link_reply(self, origin_message_id: str, reply_message_id: str) -> None:
        if not origin_message_id or not reply_message_id:
            return
        with self._lock:
            now = time.time()
            rec = self._origins.get(origin_message_id, (None, now))[0]
            if rec is None:
                rec = {
                    "origin_message_id": origin_message_id,
                    "request_id": None,
                    "chat_id": None,
                    "project_id": None,
                    "chat_type": None,
                    "sender_id": None,
                    "tenant_key": None,
                    "reply_message_ids": [],
                    "task_run_ids": [],
                }
            if reply_message_id not in rec["reply_message_ids"]:
                rec["reply_message_ids"].append(reply_message_id)
            self._reply_to_origin[reply_message_id] = origin_message_id
            self._origins[origin_message_id] = (rec, now)
            self._cleanup_unlocked(now)

    def link_task(self, origin_message_id: str, run_id: str) -> None:
        if not origin_message_id or not run_id:
            return
        with self._lock:
            now = time.time()
            rec = self._origins.get(origin_message_id, (None, now))[0]
            if rec is None:
                rec = {
                    "origin_message_id": origin_message_id,
                    "request_id": None,
                    "chat_id": None,
                    "project_id": None,
                    "chat_type": None,
                    "sender_id": None,
                    "tenant_key": None,
                    "reply_message_ids": [],
                    "task_run_ids": [],
                }
            if run_id not in rec["task_run_ids"]:
                rec["task_run_ids"].append(run_id)
            self._run_to_origin[run_id] = origin_message_id
            self._origins[origin_message_id] = (rec, now)
            self._cleanup_unlocked(now)

    def resolve_origin(
        self,
        *,
        origin_message_id: Optional[str] = None,
        reply_message_id: Optional[str] = None,
        run_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        with self._lock:
            if origin_message_id:
                return origin_message_id
            if reply_message_id and reply_message_id in self._reply_to_origin:
                return self._reply_to_origin.get(reply_message_id)
            if run_id and run_id in self._run_to_origin:
                return self._run_to_origin.get(run_id)
            if request_id and request_id in self._request_to_origin:
                return self._request_to_origin.get(request_id)
            return None

    def query(self, key: str) -> Optional[dict]:
        """按 origin/reply/run_id/request_id 任意一种 key 查询关联信息。"""
        if not key:
            return None
        # NOTE: when caller passes an arbitrary key, we must NOT treat it as origin first,
        # otherwise reply/run/request ids will short-circuit incorrectly.
        origin = self.resolve_origin(reply_message_id=key, run_id=key, request_id=key) or key
        with self._lock:
            item = self._origins.get(origin)
            if not item:
                return None
            rec, ts = item
            if time.time() - ts > self._ttl:
                self._evict_origin_unlocked(origin)
                return None
            # shallow copy for safety
            return {
                **rec,
                "updated_at": ts,
            }

    def _evict_origin_unlocked(self, origin_message_id: str) -> None:
        item = self._origins.pop(origin_message_id, None)
        if not item:
            return
        rec, _ = item
        req = rec.get("request_id")
        if req and self._request_to_origin.get(req) == origin_message_id:
            self._request_to_origin.pop(req, None)
        for rid in rec.get("reply_message_ids", []) or []:
            if self._reply_to_origin.get(rid) == origin_message_id:
                self._reply_to_origin.pop(rid, None)
        for run_id in rec.get("task_run_ids", []) or []:
            if self._run_to_origin.get(run_id) == origin_message_id:
                self._run_to_origin.pop(run_id, None)

    def _cleanup_unlocked(self, now: float) -> None:
        # TTL cleanup (OrderedDict keeps insertion order; we still do best-effort scan)
        expired = []
        for mid, (_, ts) in self._origins.items():
            if now - ts > self._ttl:
                expired.append(mid)
            else:
                break
        for mid in expired:
            self._evict_origin_unlocked(mid)
        while len(self._origins) > self._max_size:
            mid, _ = self._origins.popitem(last=False)
            self._evict_origin_unlocked(mid)
