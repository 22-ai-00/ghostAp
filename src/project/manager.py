import fcntl
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from ..card.themes import get_available_themes
from ..config import get_settings
from ..utils.errors import get_error_detail
from ..utils.lock_order import LockLevel, ordered_rlock
from .context import ADD_CHAT_ID_REJECTED, ProjectContext, ProjectStatus

logger = logging.getLogger(__name__)


class ProjectManager:
    def __init__(self, storage_path: Optional[str] = None):
        self._projects: dict[str, ProjectContext] = {}
        self._active_project: dict[str, str] = {}
        # Reverse index: bound_chat_id -> project_id, maintained by _rebuild_bound_chat_index
        # which is invoked from _save_projects and _load_projects. Any code path that mutates
        # ProjectContext.bound_chat_id MUST eventually call _save_projects to keep this in sync.
        self._bound_chat_index: dict[str, str] = {}
        self._quarantined_bound_chat_ids: set[str] = set()
        self._lock = ordered_rlock(LockLevel.PROJECT_MANAGER, name="ProjectManager._lock")
        self._color_index = 0

        # Fire-and-forget callback invoked on LRU eviction.
        # Signature: on_eviction(evicted_chat_id: str, project_name: str, project_id: str)
        self.on_eviction: Optional[Callable[[str, str, str], None]] = None

        # Optional ModeManager reference — used to clean up stale
        # _project_modes entries when a chat_id is LRU-evicted.
        self.mode_manager: Any = None

        if storage_path:
            self._storage_path = Path(storage_path)
        else:
            self._storage_path = Path.home() / ".ghostap" / "projects.json"
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_projects()

    @contextmanager
    def _file_lock(self, exclusive: bool):
        lock_path = Path(f"{self._storage_path}.lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _write_atomic(self, payload: dict):
        tmp_path = Path(f"{self._storage_path}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._storage_path)
            try:
                dir_fd = os.open(self._storage_path.parent, os.O_DIRECTORY)
            except OSError:
                return
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.debug("failed to delete temp file", exc_info=True)

    def _get_next_theme(self) -> tuple[str, str]:
        # 使用 get_available_themes() 获取非深色主题列表进行自动分配
        available_themes = get_available_themes(include_dark=False)
        theme_list = list(available_themes.values())
        theme = theme_list[self._color_index % len(theme_list)]
        self._color_index += 1
        return theme.color, theme.emoji

    @staticmethod
    def generate_id(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def create_project(
        self,
        project_id: Optional[str],
        project_name: str,
        root_path: str,
        chat_id: Optional[str] = None,
    ) -> tuple[bool, str, Optional[ProjectContext]]:
        with self._lock:
            if not project_id:
                project_id = self.generate_id(project_name)

            if project_id in self._projects:
                return False, f"项目 {project_id} 已存在", None

            expanded_path = os.path.expanduser(root_path)
            if not os.path.isdir(expanded_path):
                try:
                    os.makedirs(expanded_path, exist_ok=True)
                except Exception as e:
                    return False, f"无法创建目录 {expanded_path}: {get_error_detail(e)}", None

            previous_color_index = self._color_index
            theme_color, emoji_prefix = self._get_next_theme()
            settings = get_settings()
            yolo_enabled = bool(getattr(settings, "ttadk_yolo_default_enabled", False))

            ctx = ProjectContext(
                project_id=project_id,
                project_name=project_name,
                root_path=expanded_path,
                working_dir=expanded_path,
                status=ProjectStatus.ACTIVE,
                theme_color=theme_color,
                emoji_prefix=emoji_prefix,
                ttadk_yolo_enabled=yolo_enabled,
                owner_chat_id=chat_id or "",
                allowed_chat_ids=OrderedDict([(chat_id, time.time())]) if chat_id else OrderedDict(),
            )

            self._projects[project_id] = ctx

            if chat_id:
                self._active_project[chat_id] = project_id

            if not self._save_projects():
                self._projects.pop(project_id, None)
                if chat_id and self._active_project.get(chat_id) == project_id:
                    self._active_project.pop(chat_id, None)
                self._color_index = previous_color_index
                self._rebuild_bound_chat_index()
                return False, f"项目 {project_name} 持久化失败", None
            return True, f"项目 {project_name} 创建成功", ctx

    def get_project_for_diagnostics(self, project_id: str) -> Optional[ProjectContext]:
        """Get a project WITHOUT chat-scoped visibility check.

        **For diagnostics and system-internal use only.**
        Must NOT be called from user-facing handler code — use
        ``get_project_for_chat`` instead for chat-scoped access.
        """
        with self._lock:
            return self._projects.get(project_id)

    def get_project_for_chat(self, project_id: str, chat_id: Optional[str] = None) -> Optional[ProjectContext]:
        """Get a project with chat-scoped visibility check.

        Returns the project if it exists and is visible to *chat_id*,
        otherwise ``None``.  This is the safe entry point for card-action
        handlers where *project_id* comes from an untrusted payload
        (e.g. a card forwarded to another chat).
        """
        with self._lock:
            ctx = self._projects.get(project_id)
            if ctx is None:
                return None
            if not self._is_visible(ctx, chat_id):
                return None
            return ctx

    def persist_project_context(self, project: ProjectContext) -> bool:
        """Flush one still-registered project after an in-place state mutation."""
        with self._lock:
            if self._projects.get(project.project_id) is not project:
                return False
            return self._save_projects()

    def commit_acp_programming_activation(
        self,
        project: ProjectContext,
        *,
        tool_name: str,
        model_name: Optional[str],
        session_id: str,
        query_count: int = 0,
        activate_mode: bool = True,
    ) -> bool:
        """Persist one successful ACP selection and its project-mode snapshot.

        Startup runs before this method.  Therefore an unavailable backend, a
        rejected scheduler task, or a stale selection cannot partially replace
        the prior tool, model, mode flags, or resumable snapshot.
        """
        with self._lock:
            if self._projects.get(project.project_id) is not project:
                return False
            previous = project.commit_acp_programming_activation(
                tool_name,
                tool_name,
                model_name,
                session_id,
                query_count,
                activate_mode,
            )
            if self._save_projects():
                return True
            project.restore_acp_programming_activation(previous)
            return False

    @staticmethod
    def _is_visible(ctx: ProjectContext, chat_id: Optional[str]) -> bool:
        """Check if a project is visible to the given chat_id.

        Visibility rules:
        - If chat_id is None (no filter), always visible.
        - If allowed_chat_ids is empty (legacy project), visible to all.
        - Otherwise, chat_id must be in allowed_chat_ids.
        """
        if chat_id is None:
            return True
        if not ctx.allowed_chat_ids:
            return True
        return chat_id in ctx.allowed_chat_ids

    def get_all_projects(self, sort_by_recent: bool = True, chat_id: Optional[str] = None) -> list[ProjectContext]:
        with self._lock:
            snapshot = list(self._projects.values())
        projects = [p for p in snapshot if self._is_visible(p, chat_id)]
        if sort_by_recent:
            projects.sort(key=lambda p: p.last_active, reverse=True)
        return projects

    def find_project_by_path(self, path: str, chat_id: Optional[str] = None) -> Optional[ProjectContext]:
        expanded = os.path.expanduser(os.path.abspath(path))
        with self._lock:
            snapshot = list(self._projects.values())
        for ctx in snapshot:
            if ctx.root_path == expanded and self._is_visible(ctx, chat_id):
                return ctx
        return None

    def get_or_create_project_for_path(
        self,
        path: str,
        chat_id: Optional[str] = None,
    ) -> tuple[ProjectContext, bool]:
        expanded = os.path.expanduser(os.path.abspath(path))

        existing = self.find_project_by_path(expanded, chat_id=chat_id)
        if existing:
            if chat_id:
                self.set_active_project(chat_id, existing.project_id)
            else:
                existing.touch()
                self._save_projects()
            return existing, False

        basename = os.path.basename(expanded.rstrip(os.sep))
        if not basename:
            basename = "root"

        project_id = self.generate_id(basename)
        original_id = project_id
        counter = 1
        while project_id in self._projects:
            project_id = f"{original_id}_{counter}"
            counter += 1

        success, msg, ctx = self.create_project(
            project_id=project_id,
            project_name=basename,
            root_path=expanded,
            chat_id=chat_id,
        )

        if success and ctx:
            return ctx, True
        else:
            raise RuntimeError(f"创建项目失败: {msg}")

    def search_projects(self, query: str, chat_id: Optional[str] = None) -> list[ProjectContext]:
        query_lower = query.lower()
        with self._lock:
            snapshot = list(self._projects.values())
        results = []
        for ctx in snapshot:
            if not self._is_visible(ctx, chat_id):
                continue
            if (
                query_lower in ctx.project_name.lower()
                or query_lower in ctx.project_id.lower()
                or query_lower in ctx.root_path.lower()
            ):
                results.append(ctx)
        results.sort(key=lambda p: p.last_active, reverse=True)
        return results

    def validate_project_path(self, project_id: str, chat_id: Optional[str] = None) -> tuple[bool, str]:
        with self._lock:
            ctx = self._projects.get(project_id)
            if not ctx:
                return False, f"项目 {project_id} 不存在"

            if chat_id is not None and not self._is_visible(ctx, chat_id):
                return False, "无权访问该项目"

            if not os.path.isdir(ctx.root_path):
                return False, f"项目路径不存在: {ctx.root_path}"

            return True, ctx.root_path

    def get_active_project(self, chat_id: str) -> Optional[ProjectContext]:
        with self._lock:
            project_id = self._active_project.get(chat_id)
            ctx = self._projects.get(project_id) if project_id else None
            if ctx and not self._is_visible(ctx, chat_id):
                return None
            return ctx

    def set_active_project(self, chat_id: str, project_id: str) -> tuple[bool, str]:
        eviction_info: tuple[str, str, str] | None = None  # (evicted_chat_id, project_name, project_id)
        with self._lock:
            if project_id not in self._projects:
                return False, f"项目 {project_id} 不存在"

            ctx = self._projects[project_id]

            # Legacy project backfill: inject chat_id into empty allowed_chat_ids
            # so isolation gradually takes effect for pre-upgrade projects.
            if not ctx.allowed_chat_ids and chat_id:
                ctx.owner_chat_id = ctx.owner_chat_id or chat_id
                logger.info(
                    "Legacy backfill: injecting chat_id=%s into project=%s",
                    chat_id[:12], project_id,
                )

            old_project_id = self._active_project.get(chat_id)
            if old_project_id == project_id and ctx.status == ProjectStatus.ACTIVE:
                refresh_result = ctx.add_chat_id(chat_id)  # Refresh LRU timestamp
                if refresh_result == ADD_CHAT_ID_REJECTED:
                    return False, f"项目 {ctx.project_name} 的群绑定数已满，无法关联当前群"
                ctx.touch()
                self._save_projects()
                return True, f"已切换到项目 {ctx.project_name}"

            old_status = None
            if old_project_id and old_project_id in self._projects:
                old_ctx = self._projects[old_project_id]
                old_status = old_ctx.status
                if old_ctx.status == ProjectStatus.ACTIVE:
                    old_ctx.status = ProjectStatus.IDLE

            self._active_project[chat_id] = project_id
            ctx.status = ProjectStatus.ACTIVE
            evicted = ctx.add_chat_id(chat_id)
            if evicted == ADD_CHAT_ID_REJECTED:
                # Rollback: undo the _active_project write
                if old_project_id is not None:
                    self._active_project[chat_id] = old_project_id
                else:
                    self._active_project.pop(chat_id, None)
                # Restore old project status to its original value
                if old_project_id and old_project_id in self._projects:
                    old_ctx = self._projects[old_project_id]
                    old_ctx.status = old_status
                ctx.status = ProjectStatus.IDLE
                return False, f"项目 {ctx.project_name} 的群绑定数已满，无法关联当前群"
            if evicted:
                logger.warning(
                    "LRU eviction: chat=%s removed from project=%s (project_name=%s) "
                    "due to allowed_chat_ids capacity limit",
                    evicted[:12], project_id, ctx.project_name,
                )
                # Capture eviction info — callback fires AFTER lock release (F-01).
                eviction_info = (evicted, ctx.project_name, project_id)
                # Clean up orphan _active_project entry INSIDE the lock (Q-30-1 fix).
                # Conditional pop: only remove if the entry still points to the
                # current project_id.  Another thread may have already re-bound
                # the evicted chat_id to a different project between add_chat_id
                # and this line; unconditional pop would clobber that new binding.
                if self._active_project.get(evicted) == project_id:
                    self._active_project.pop(evicted, None)
            ctx.touch()

            self._save_projects()
            result = True, f"已切换到项目 {ctx.project_name}"

        # Fire eviction callback OUTSIDE the lock to avoid blocking (F-01).
        if eviction_info:
            evicted_cid, ev_proj_name, ev_proj_id = eviction_info
            # Clean up stale ModeManager entries for the evicted chat (AC-R01).
            if self.mode_manager is not None:
                try:
                    self.mode_manager.clear_modes_for_chat(evicted_cid)
                except Exception as mm_err:
                    logger.warning("mode_manager.clear_modes_for_chat failed: %s", mm_err)
            if self.on_eviction:
                try:
                    self.on_eviction(evicted_cid, ev_proj_name, ev_proj_id)
                except Exception as cb_err:
                    logger.warning("on_eviction callback failed: %s", cb_err)

        return result

    def close_project(self, project_id: str) -> tuple[bool, str]:
        with self._lock:
            if project_id not in self._projects:
                return False, f"项目 {project_id} 不存在"

            ctx = self._projects[project_id]
            previous_status = ctx.status
            removed_active = {
                chat_id: active_id
                for chat_id, active_id in self._active_project.items()
                if active_id == project_id
            }
            ctx.status = ProjectStatus.CLOSED

            for chat_id, active_id in list(self._active_project.items()):
                if active_id == project_id:
                    del self._active_project[chat_id]

            del self._projects[project_id]
            if not self._save_projects():
                ctx.status = previous_status
                self._projects[project_id] = ctx
                self._active_project.update(removed_active)
                self._rebuild_bound_chat_index()
                return False, f"项目 {ctx.project_name} 关闭持久化失败"
            return True, f"项目 {ctx.project_name} 已关闭"

    def update_working_dir(self, project_id: str, new_dir: str) -> tuple[bool, str]:
        with self._lock:
            ctx = self._projects.get(project_id)
            if not ctx:
                return False, f"项目 {project_id} 不存在"

            expanded = os.path.expanduser(new_dir)
            if not os.path.isabs(expanded):
                expanded = os.path.normpath(os.path.join(ctx.working_dir, expanded))

            if not os.path.isdir(expanded):
                return False, f"目录不存在: {expanded}"

            ctx.working_dir = expanded
            ctx.touch()
            self._save_projects()
            return True, expanded

    def find_project_by_name(self, name: str, chat_id: Optional[str] = None) -> Optional[ProjectContext]:
        name_lower = name.lower()
        with self._lock:
            snapshot = list(self._projects.values())
        for ctx in snapshot:
            if not self._is_visible(ctx, chat_id):
                continue
            if ctx.project_name.lower() == name_lower or ctx.project_id.lower() == name_lower:
                return ctx
        for ctx in snapshot:
            if not self._is_visible(ctx, chat_id):
                continue
            if name_lower in ctx.project_name.lower() or name_lower in ctx.project_id.lower():
                return ctx
        return None

    def find_project_for_owner_control(
        self,
        reference: str,
    ) -> Optional[ProjectContext]:
        """Resolve only an exact project ID or one unambiguous exact name."""

        if not isinstance(reference, str) or not reference:
            return None
        with self._lock:
            by_id = self._projects.get(reference)
            if by_id is not None:
                return by_id
            matches = [
                project
                for project in self._projects.values()
                if project.project_name == reference
            ]
            return matches[0] if len(matches) == 1 else None

    def bind_managed_chat(
        self,
        project_id: str,
        chat_id: str,
        *,
        chat_name: str = "",
        created_at: float,
    ) -> bool:
        """Persist a validated Owner-control group binding."""

        with self._lock:
            project = self._projects.get(project_id)
            if project is None or not chat_id:
                return False
            indexed = self._bound_chat_index.get(chat_id)
            if indexed not in (None, project_id):
                return False
            snapshot = (
                project.bound_chat_id,
                project.bound_chat_name,
                project.bound_chat_created_at,
                OrderedDict(project.allowed_chat_ids),
            )
            project.bound_chat_id = chat_id
            project.bound_chat_name = chat_name
            project.bound_chat_created_at = created_at
            project.add_chat_id(chat_id)
            self._quarantined_bound_chat_ids.discard(chat_id)
            if self._save_projects():
                return True
            (
                project.bound_chat_id,
                project.bound_chat_name,
                project.bound_chat_created_at,
                project.allowed_chat_ids,
            ) = snapshot
            self._rebuild_bound_chat_index()
            return False

    def revoke_managed_chat(self, chat_id: str) -> bool:
        """Clear a Project binding after durable Registry revocation."""

        with self._lock:
            project = next(
                (
                    item
                    for item in self._projects.values()
                    if item.bound_chat_id == chat_id
                ),
                None,
            )
            if project is None:
                self._bound_chat_index.pop(chat_id, None)
                return True
            snapshot = (
                project.bound_chat_id,
                project.bound_chat_name,
                project.bound_chat_created_at,
                OrderedDict(project.allowed_chat_ids),
            )
            project.bound_chat_id = ""
            project.bound_chat_name = ""
            project.bound_chat_created_at = 0.0
            project.allowed_chat_ids.pop(chat_id, None)
            if self._save_projects():
                return True
            (
                project.bound_chat_id,
                project.bound_chat_name,
                project.bound_chat_created_at,
                project.allowed_chat_ids,
            ) = snapshot
            self._rebuild_bound_chat_index()
            return False

    def find_project_by_name_with_hint(
        self, name: str, chat_id: Optional[str] = None
    ) -> tuple[Optional[ProjectContext], Optional[str]]:
        """Like ``find_project_by_name`` but returns a hint when a project exists
        globally but is not visible to *chat_id*.

        Returns ``(project, None)`` on success or ``(None, hint_message)`` when
        the project exists in another chat's scope.
        """
        result = self.find_project_by_name(name, chat_id=chat_id)
        if result is not None:
            return result, None
        # Check globally (no chat filter) to detect cross-chat case
        if chat_id is not None:
            global_result = self.find_project_by_name(name, chat_id=None)
            if global_result is not None:
                # Disambiguate: was the chat previously bound but evicted by LRU?
                if chat_id in global_result.evicted_chat_ids:
                    return None, (
                        "该项目因关联群数达上限已自动解绑，"
                        "如需重新关联，请使用 /new 创建新项目"
                    )
                return None, "该项目已绑定到其他群聊，如需在当前群使用同一仓库，请使用 /new 创建新项目"
        return None, None

    def _save_projects(self) -> bool:
        try:
            data = {
                "projects": {pid: ctx.to_snapshot() for pid, ctx in self._projects.items()},
                "active_project": self._active_project,
                "color_index": self._color_index,
            }
            with self._file_lock(True):
                self._write_atomic(data)
            self._rebuild_bound_chat_index()
            return True
        except Exception as e:
            logger.error("保存项目数据失败: %s", get_error_detail(e))
            return False

    def _rebuild_bound_chat_index(self):
        """Rebuild the reverse index (bound_chat_id -> project_id).

        Called from _save_projects and _load_projects; must hold self._lock
        (both callers already do, since _save_projects is only called from
        locked sections and _load_projects runs during __init__).
        """
        index: dict[str, str] = {}
        for pid, ctx in self._projects.items():
            if (
                ctx.bound_chat_id
                and ctx.bound_chat_id not in self._quarantined_bound_chat_ids
            ):
                # Last-write wins on duplicate bound_chat_id (should be unique by design,
                # but defensive in case of inconsistent state)
                index[ctx.bound_chat_id] = pid
        self._bound_chat_index = index

    def quarantine_bound_chat(self, chat_id: str) -> None:
        """Fail closed when compensating persistence cannot be confirmed."""

        if not chat_id:
            return
        with self._lock:
            self._quarantined_bound_chat_ids.add(chat_id)
            self._bound_chat_index.pop(chat_id, None)

    def find_by_bound_chat_id(self, chat_id: str) -> Optional[ProjectContext]:
        """Return the project whose bound_chat_id equals *chat_id*, if any.

        Used to detect "project chat" groups created via /new-chat, so the
        dispatcher can route free-form text directly into Coco programming mode.
        Returns None when no project is bound to this chat.
        """
        if not chat_id:
            return None
        with self._lock:
            project_id = self._bound_chat_index.get(chat_id)
            if not project_id:
                return None
            return self._projects.get(project_id)

    def _load_projects(self):
        if not self._storage_path.exists():
            return

        try:
            with self._file_lock(True):
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            for pid, snap in data.get("projects", {}).items():
                try:
                    ctx = ProjectContext.from_snapshot(snap)
                    if ctx.status == ProjectStatus.CLOSED:
                        continue
                    ctx.status = ProjectStatus.IDLE
                    self._projects[pid] = ctx
                except Exception as e:
                    logger.error("加载项目 %s 失败: %s", pid, get_error_detail(e))

            self._active_project = data.get("active_project", {})
            self._color_index = data.get("color_index", 0)
            self._rebuild_bound_chat_index()
        except Exception as e:
            corrupt_path = Path(f"{self._storage_path}.corrupt.{int(time.time())}")
            try:
                if self._storage_path.exists():
                    os.replace(self._storage_path, corrupt_path)
                    logger.error("加载项目数据失败，已备份损坏文件到: %s", corrupt_path)
            except Exception:
                logger.error("加载项目数据失败: %s", get_error_detail(e))
