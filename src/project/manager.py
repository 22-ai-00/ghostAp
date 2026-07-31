import fcntl
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..card.themes import get_available_themes
from ..config import get_settings
from ..utils.errors import get_error_detail
from ..utils.lock_order import LockLevel, ordered_rlock
from .context import ADD_CHAT_ID_REJECTED, ProjectContext, ProjectStatus

logger = logging.getLogger(__name__)


class ProjectCommitUncertainError(OSError):
    """A Project snapshot replace committed but parent durability is unknown."""

    def __init__(self, message: str, *, committed: bool) -> None:
        super().__init__(message)
        self.committed = committed


@dataclass(frozen=True, slots=True)
class ManagedChatBindingSnapshot:
    bound_chat_id: str
    bound_chat_name: str
    bound_chat_created_at: float
    allowed_chat_ids: tuple[tuple[str, float], ...]
    owner_chat_id: str = ""
    project_name: str = ""
    binding_generation: int = 0


@dataclass(frozen=True, slots=True)
class ManagedChatBindingSaga:
    operation_id: str
    project_id: str
    chat_id: str
    snapshot: ManagedChatBindingSnapshot
    expected: ManagedChatBindingSnapshot | None
    expected_origin: str = ""
    expected_owner_id: str = ""
    expected_receiving_bot_ref: str = ""
    expected_root_ref: str = ""
    remove_project_on_restore: bool = False
    displaced_legacy_operation_id: str = ""


class ProjectManager:
    def __init__(self, storage_path: Optional[str] = None):
        self._projects: dict[str, ProjectContext] = {}
        self._active_project: dict[str, str] = {}
        # Reverse index: bound_chat_id -> project_id, maintained by _rebuild_bound_chat_index
        # which is invoked from _save_projects and _load_projects. Any code path that mutates
        # ProjectContext.bound_chat_id MUST eventually call _save_projects to keep this in sync.
        self._bound_chat_index: dict[str, str] = {}
        self._quarantined_bound_chat_ids: set[str] = set()
        self._managed_group_residuals: dict[str, tuple[str, str]] = {}
        self._managed_chat_binding_sagas: dict[str, ManagedChatBindingSaga] = {}
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
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        replaced = False
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._storage_path)
            replaced = True
            dir_fd = os.open(self._storage_path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception as exc:
            if replaced:
                raise ProjectCommitUncertainError(
                    "project snapshot parent durability is uncertain",
                    committed=True,
                ) from exc
            raise
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

    def create_project_with_managed_chat_saga(
        self,
        *,
        project_id: Optional[str],
        project_name: str,
        root_path: str,
        owner_chat_id: str,
        managed_chat_id: str,
        managed_chat_name: str,
        created_at: float,
        operation_id: str,
        expected_origin: object,
        expected_owner_id: str,
        expected_receiving_bot_ref: str,
        additional_chat_id: str = "",
    ) -> tuple[bool, str, Optional[ProjectContext]]:
        """Create, bind, and prepare one removal-aware saga in one snapshot."""

        with self._lock:
            resolved_id = project_id or self.generate_id(project_name)
            if resolved_id in self._projects:
                return False, f"项目 {resolved_id} 已存在", None
            if any(
                saga.project_id == resolved_id
                for saga in self._managed_chat_binding_sagas.values()
            ):
                return False, f"项目 {resolved_id} 已有未完成的群绑定事务", None
            expanded_path = os.path.expanduser(root_path)
            if not os.path.isdir(expanded_path):
                try:
                    os.makedirs(expanded_path, exist_ok=True)
                except Exception as exc:
                    return (
                        False,
                        f"无法创建目录 {expanded_path}: {get_error_detail(exc)}",
                        None,
                    )
            previous_color_index = self._color_index
            theme_color, emoji_prefix = self._get_next_theme()
            settings = get_settings()
            ctx = ProjectContext(
                project_id=resolved_id,
                project_name=project_name,
                root_path=expanded_path,
                working_dir=expanded_path,
                status=ProjectStatus.ACTIVE,
                theme_color=theme_color,
                emoji_prefix=emoji_prefix,
                ttadk_yolo_enabled=bool(
                    getattr(settings, "ttadk_yolo_default_enabled", False)
                ),
                owner_chat_id=owner_chat_id,
                allowed_chat_ids=(
                    OrderedDict([(owner_chat_id, time.time())])
                    if owner_chat_id
                    else OrderedDict()
                ),
            )
            before = self._binding_snapshot(ctx)
            ctx.bound_chat_id = managed_chat_id
            ctx.bound_chat_name = managed_chat_name
            ctx.bound_chat_created_at = created_at
            ctx.managed_binding_generation += 1
            ctx.add_chat_id(managed_chat_id)
            if additional_chat_id and additional_chat_id != managed_chat_id:
                ctx.add_chat_id(additional_chat_id)
            expected = self._binding_snapshot(ctx)
            saga = ManagedChatBindingSaga(
                operation_id=operation_id,
                project_id=resolved_id,
                chat_id=managed_chat_id,
                snapshot=before,
                expected=expected,
                expected_origin=self._origin_value(expected_origin),
                expected_owner_id=expected_owner_id,
                expected_receiving_bot_ref=expected_receiving_bot_ref,
                expected_root_ref=expanded_path,
                remove_project_on_restore=True,
            )
            self._projects[resolved_id] = ctx
            self._active_project[managed_chat_id] = resolved_id
            self._managed_chat_binding_sagas[operation_id] = saga
            try:
                saved = self._save_projects()
            except ProjectCommitUncertainError:
                raise
            if not saved:
                self._projects.pop(resolved_id, None)
                self._active_project.pop(managed_chat_id, None)
                self._managed_chat_binding_sagas.pop(operation_id, None)
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

    @staticmethod
    def _origin_value(origin: object) -> str:
        return str(getattr(origin, "value", origin) or "")

    @staticmethod
    def _binding_snapshot(project: ProjectContext) -> ManagedChatBindingSnapshot:
        return ManagedChatBindingSnapshot(
            bound_chat_id=project.bound_chat_id,
            bound_chat_name=project.bound_chat_name,
            bound_chat_created_at=project.bound_chat_created_at,
            allowed_chat_ids=tuple(project.allowed_chat_ids.items()),
            owner_chat_id=project.owner_chat_id,
            project_name=project.project_name,
            binding_generation=project.managed_binding_generation,
        )

    @classmethod
    def _saga_matches_project(
        cls,
        saga: ManagedChatBindingSaga,
        project: ProjectContext,
    ) -> bool:
        return (
            saga.expected is not None
            and bool(saga.expected_root_ref)
            and project.root_path == saga.expected_root_ref
            and cls._binding_snapshot(project) == saga.expected
        )

    def _quarantine_saga_locked(
        self,
        saga: ManagedChatBindingSaga,
        project: ProjectContext | None,
    ) -> bool:
        self._quarantined_bound_chat_ids.add(saga.chat_id)
        if project is not None and project.bound_chat_id:
            self._quarantined_bound_chat_ids.add(project.bound_chat_id)
        self._save_projects()
        self._rebuild_bound_chat_index()
        return False

    def bind_managed_chat(
        self,
        project_id: str,
        chat_id: str,
        *,
        chat_name: str = "",
        created_at: float,
    ) -> bool:
        """Persist a validated Owner-control group binding."""

        bound, _ = self.bind_managed_chat_for_saga(
            project_id,
            chat_id,
            chat_name=chat_name,
            created_at=created_at,
        )
        return bound

    def bind_managed_chat_for_saga(
        self,
        project_id: str,
        chat_id: str,
        *,
        chat_name: str = "",
        created_at: float,
        operation_id: str = "",
        additional_chat_id: str = "",
        project_name: str = "",
        remove_project_on_restore: bool = False,
        expected_origin: object = "",
        expected_owner_id: str = "",
        expected_receiving_bot_ref: str = "",
        replace_legacy_saga: bool = False,
    ) -> tuple[bool, ManagedChatBindingSnapshot | None]:
        """Bind and return the durable pre-bind state for compensation."""

        with self._lock:
            project = self._projects.get(project_id)
            if project is None or not chat_id:
                return False, None
            indexed = self._bound_chat_index.get(chat_id)
            if indexed not in (None, project_id):
                return False, None
            existing_saga = self._managed_chat_binding_sagas.get(operation_id)
            replaced_legacy_saga = None
            if existing_saga is not None:
                if (
                    existing_saga.project_id != project_id
                    or existing_saga.chat_id != chat_id
                    or existing_saga.remove_project_on_restore
                    is not remove_project_on_restore
                ):
                    return False, None
                if not self._saga_matches_project(existing_saga, project):
                    self._quarantine_saga_locked(existing_saga, project)
                    return False, None
                return True, existing_saga.snapshot
            else:
                project_sagas = tuple(
                    saga
                    for saga in self._managed_chat_binding_sagas.values()
                    if saga.project_id == project_id
                    and saga.operation_id != operation_id
                )
                if project_sagas:
                    if (
                        replace_legacy_saga
                        and len(project_sagas) == 1
                        and project_sagas[0].expected is None
                        and project_sagas[0].chat_id == chat_id
                        and project_sagas[0].expected_root_ref == project.root_path
                        and self._managed_group_residuals.get(
                            project_sagas[0].operation_id
                        )
                        == (chat_id, "legacy_saga_resolution_required")
                        and chat_id in self._quarantined_bound_chat_ids
                    ):
                        replaced_legacy_saga = project_sagas[0]
                    else:
                        return False, None
                snapshot = self._binding_snapshot(project)
            project.bound_chat_id = chat_id
            project.bound_chat_name = chat_name
            project.bound_chat_created_at = created_at
            project.managed_binding_generation += 1
            if project_name:
                project.project_name = project_name
            project.add_chat_id(chat_id)
            if additional_chat_id and additional_chat_id != chat_id:
                project.add_chat_id(additional_chat_id)
            expected = self._binding_snapshot(project)
            if operation_id:
                self._managed_chat_binding_sagas[operation_id] = (
                    ManagedChatBindingSaga(
                        operation_id=operation_id,
                        project_id=project_id,
                        chat_id=chat_id,
                        snapshot=snapshot,
                        expected=expected,
                        expected_origin=self._origin_value(expected_origin),
                        expected_owner_id=expected_owner_id,
                        expected_receiving_bot_ref=expected_receiving_bot_ref,
                        expected_root_ref=project.root_path,
                        remove_project_on_restore=remove_project_on_restore,
                        displaced_legacy_operation_id=(
                            replaced_legacy_saga.operation_id
                            if replaced_legacy_saga is not None
                            else ""
                        ),
                    )
                )
            was_quarantined = chat_id in self._quarantined_bound_chat_ids
            if replaced_legacy_saga is None:
                self._quarantined_bound_chat_ids.discard(chat_id)
            try:
                if self._save_projects():
                    return True, snapshot
            except ProjectCommitUncertainError:
                raise
            project.bound_chat_id = snapshot.bound_chat_id
            project.bound_chat_name = snapshot.bound_chat_name
            project.bound_chat_created_at = snapshot.bound_chat_created_at
            project.allowed_chat_ids = OrderedDict(snapshot.allowed_chat_ids)
            project.owner_chat_id = snapshot.owner_chat_id
            project.project_name = snapshot.project_name
            project.managed_binding_generation = snapshot.binding_generation
            if was_quarantined:
                self._quarantined_bound_chat_ids.add(chat_id)
            if operation_id and existing_saga is None:
                self._managed_chat_binding_sagas.pop(operation_id, None)
            self._rebuild_bound_chat_index()
            return False, None

    def restore_managed_chat_binding(
        self,
        project_id: str,
        snapshot: ManagedChatBindingSnapshot,
        operation_id: str = "",
    ) -> bool:
        """Durably compensate a prepared managed-chat binding."""

        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return False
            current = ManagedChatBindingSnapshot(
                bound_chat_id=project.bound_chat_id,
                bound_chat_name=project.bound_chat_name,
                bound_chat_created_at=project.bound_chat_created_at,
                allowed_chat_ids=tuple(project.allowed_chat_ids.items()),
                owner_chat_id=project.owner_chat_id,
                project_name=project.project_name,
                binding_generation=project.managed_binding_generation,
            )
            project.bound_chat_id = snapshot.bound_chat_id
            project.bound_chat_name = snapshot.bound_chat_name
            project.bound_chat_created_at = snapshot.bound_chat_created_at
            project.allowed_chat_ids = OrderedDict(snapshot.allowed_chat_ids)
            project.owner_chat_id = snapshot.owner_chat_id
            project.project_name = snapshot.project_name
            project.managed_binding_generation = snapshot.binding_generation
            old_saga = self._managed_chat_binding_sagas.pop(operation_id, None)
            if self._save_projects():
                return True
            project.bound_chat_id = current.bound_chat_id
            project.bound_chat_name = current.bound_chat_name
            project.bound_chat_created_at = current.bound_chat_created_at
            project.allowed_chat_ids = OrderedDict(current.allowed_chat_ids)
            project.owner_chat_id = current.owner_chat_id
            project.project_name = current.project_name
            project.managed_binding_generation = current.binding_generation
            if old_saga is not None:
                self._managed_chat_binding_sagas[operation_id] = old_saga
            self._rebuild_bound_chat_index()
            return False

    def pending_managed_chat_binding_sagas(
        self,
    ) -> tuple[ManagedChatBindingSaga, ...]:
        with self._lock:
            return tuple(
                self._managed_chat_binding_sagas[key]
                for key in sorted(self._managed_chat_binding_sagas)
            )

    def restore_managed_chat_binding_saga(self, operation_id: str) -> bool:
        """Durably restore the exact pre-bind state for a pending saga."""

        with self._lock:
            saga = self._managed_chat_binding_sagas.get(operation_id)
            if saga is None:
                return True
            project = self._projects.get(saga.project_id)
            if project is None:
                return False
            if not self._saga_matches_project(saga, project):
                return self._quarantine_saga_locked(saga, project)
            if saga.remove_project_on_restore:
                removed_active = {
                    chat_id: active_id
                    for chat_id, active_id in self._active_project.items()
                    if active_id == saga.project_id
                }
                self._projects.pop(saga.project_id)
                for chat_id in removed_active:
                    self._active_project.pop(chat_id, None)
                self._managed_chat_binding_sagas.pop(operation_id)
                if self._save_projects():
                    return True
                self._projects[saga.project_id] = project
                self._active_project.update(removed_active)
                self._managed_chat_binding_sagas[operation_id] = saga
                self._rebuild_bound_chat_index()
                return False
            return self.restore_managed_chat_binding(
                saga.project_id,
                saga.snapshot,
                operation_id=operation_id,
            )

    def complete_managed_chat_binding_saga(self, operation_id: str) -> bool:
        if not operation_id:
            return True
        with self._lock:
            saga = self._managed_chat_binding_sagas.get(operation_id)
            if saga is None:
                return True
            project = self._projects.get(saga.project_id)
            if project is None or not self._saga_matches_project(saga, project):
                return self._quarantine_saga_locked(saga, project)
            displaced = None
            displaced_residual = None
            displaced_was_quarantined = False
            if saga.displaced_legacy_operation_id:
                displaced = self._managed_chat_binding_sagas.get(
                    saga.displaced_legacy_operation_id
                )
                displaced_residual = self._managed_group_residuals.get(
                    saga.displaced_legacy_operation_id
                )
                if (
                    displaced is None
                    or displaced.expected is not None
                    or displaced.project_id != saga.project_id
                    or displaced.chat_id != saga.chat_id
                    or displaced.expected_root_ref != saga.expected_root_ref
                    or displaced_residual
                    != (saga.chat_id, "legacy_saga_resolution_required")
                    or saga.chat_id not in self._quarantined_bound_chat_ids
                ):
                    return self._quarantine_saga_locked(saga, project)
            previous = self._managed_chat_binding_sagas.pop(operation_id)
            if displaced is not None:
                self._managed_chat_binding_sagas.pop(displaced.operation_id)
                self._managed_group_residuals.pop(displaced.operation_id)
                displaced_was_quarantined = (
                    displaced.chat_id in self._quarantined_bound_chat_ids
                )
                self._quarantined_bound_chat_ids.discard(displaced.chat_id)
            if self._save_projects():
                return True
            self._managed_chat_binding_sagas[operation_id] = previous
            if displaced is not None:
                self._managed_chat_binding_sagas[displaced.operation_id] = displaced
                if displaced_residual is not None:
                    self._managed_group_residuals[
                        displaced.operation_id
                    ] = displaced_residual
                if displaced_was_quarantined:
                    self._quarantined_bound_chat_ids.add(displaced.chat_id)
            return False

    def managed_chat_binding_saga(self, operation_id: str) -> ManagedChatBindingSaga | None:
        with self._lock:
            return self._managed_chat_binding_sagas.get(operation_id)

    def pending_managed_chat_binding_sagas_for_project(
        self,
        project_id: str,
        chat_id: str = "",
    ) -> tuple[ManagedChatBindingSaga, ...]:
        with self._lock:
            return tuple(
                saga
                for saga in self._managed_chat_binding_sagas.values()
                if saga.project_id == project_id
                and (not chat_id or saga.chat_id == chat_id)
            )

    def resolve_managed_chat_binding_saga(
        self,
        *,
        project_id: str,
        chat_id: str,
        expected_origin: object,
        expected_owner_id: str,
        expected_receiving_bot_ref: str,
        expected_root_ref: str,
    ) -> ManagedChatBindingSaga | None:
        """Resolve exactly one compatible saga independent of display name."""

        origin = self._origin_value(expected_origin)
        with self._lock:
            matches = tuple(
                saga
                for saga in self._managed_chat_binding_sagas.values()
                if saga.project_id == project_id
                and saga.chat_id == chat_id
                and saga.expected is not None
                and saga.expected_origin == origin
                and saga.expected_owner_id == expected_owner_id
                and saga.expected_receiving_bot_ref
                == expected_receiving_bot_ref
                and saga.expected_root_ref == expected_root_ref
            )
            return matches[0] if len(matches) == 1 else None

    def validate_managed_chat_binding_saga(self, operation_id: str) -> bool:
        """CAS-check one pending saga and quarantine any mismatched binding."""

        with self._lock:
            saga = self._managed_chat_binding_sagas.get(operation_id)
            if saga is None:
                return False
            project = self._projects.get(saga.project_id)
            if project is not None and self._saga_matches_project(saga, project):
                return True
            return self._quarantine_saga_locked(saga, project)

    def mark_legacy_saga_resolution_required(self, operation_id: str) -> bool:
        """Persist a fail-closed Owner-resolution marker for one legacy saga."""

        with self._lock:
            saga = self._managed_chat_binding_sagas.get(operation_id)
            if saga is None or saga.expected is not None:
                return False
            self._quarantined_bound_chat_ids.add(saga.chat_id)
            self._managed_group_residuals[operation_id] = (
                saga.chat_id,
                "legacy_saga_resolution_required",
            )
            saved = self._save_projects()
            self._rebuild_bound_chat_index()
            return saved

    def complete_exact_active_legacy_saga(self, operation_id: str) -> bool:
        """Consume a legacy saga only after its Registry facts were verified."""

        with self._lock:
            saga = self._managed_chat_binding_sagas.get(operation_id)
            if saga is None:
                return True
            project = self._projects.get(saga.project_id)
            if (
                saga.expected is not None
                or project is None
                or project.bound_chat_id != saga.chat_id
                or project.root_path != saga.expected_root_ref
            ):
                return False
            residual = self._managed_group_residuals.get(operation_id)
            self._managed_chat_binding_sagas.pop(operation_id)
            self._managed_group_residuals.pop(operation_id, None)
            was_quarantined = saga.chat_id in self._quarantined_bound_chat_ids
            self._quarantined_bound_chat_ids.discard(saga.chat_id)
            if self._save_projects():
                return True
            self._managed_chat_binding_sagas[operation_id] = saga
            if residual is not None:
                self._managed_group_residuals[operation_id] = residual
            if was_quarantined:
                self._quarantined_bound_chat_ids.add(saga.chat_id)
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

    def record_managed_group_residual(
        self,
        operation_id: str,
        chat_id: str,
        delete_state: str,
    ) -> bool:
        if (
            not operation_id
            or not chat_id
            or delete_state not in {
                "create_outcome_unknown",
                "delete_rejected",
                "delete_unknown",
                "delete_guarded",
                "recovered_chat_invalid",
                "registry_bind_uncertain",
                "untrusted_retained",
                "legacy_saga_resolution_required",
            }
        ):
            return False
        with self._lock:
            existing = self._managed_group_residuals.get(operation_id)
            value = (chat_id, delete_state)
            if existing not in (None, value):
                return False
            self._managed_group_residuals[operation_id] = value
            return self._save_projects()

    def managed_group_residual(
        self,
        operation_id: str,
    ) -> tuple[str, str] | None:
        with self._lock:
            return self._managed_group_residuals.get(operation_id)

    def consume_managed_group_residual(
        self,
        operation_id: str,
        chat_id: str,
        delete_state: str,
    ) -> bool:
        """CAS-consume one exact residual after successful reconciliation."""

        with self._lock:
            expected = (chat_id, delete_state)
            if self._managed_group_residuals.get(operation_id) != expected:
                return False
            self._managed_group_residuals.pop(operation_id)
            if self._save_projects():
                return True
            self._managed_group_residuals[operation_id] = expected
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
                "quarantined_bound_chat_ids": sorted(
                    self._quarantined_bound_chat_ids
                ),
                "managed_group_residuals": {
                    operation_id: {
                        "chat_id": value[0],
                        "delete_state": value[1],
                    }
                    for operation_id, value in sorted(
                        self._managed_group_residuals.items()
                    )
                },
                "managed_chat_binding_sagas": {
                    operation_id: {
                        "chat_id": saga.chat_id,
                        "project_id": saga.project_id,
                        "remove_project_on_restore": saga.remove_project_on_restore,
                        "snapshot": {
                            "allowed_chat_ids": list(saga.snapshot.allowed_chat_ids),
                            "bound_chat_created_at": saga.snapshot.bound_chat_created_at,
                            "bound_chat_id": saga.snapshot.bound_chat_id,
                            "bound_chat_name": saga.snapshot.bound_chat_name,
                            "owner_chat_id": saga.snapshot.owner_chat_id,
                            "project_name": saga.snapshot.project_name,
                            "binding_generation": saga.snapshot.binding_generation,
                        },
                        "expected": (
                            None
                            if saga.expected is None
                            else {
                                "allowed_chat_ids": list(
                                    saga.expected.allowed_chat_ids
                                ),
                                "bound_chat_created_at": saga.expected.bound_chat_created_at,
                                "bound_chat_id": saga.expected.bound_chat_id,
                                "bound_chat_name": saga.expected.bound_chat_name,
                                "owner_chat_id": saga.expected.owner_chat_id,
                                "project_name": saga.expected.project_name,
                                "binding_generation": saga.expected.binding_generation,
                            }
                        ),
                        "expected_origin": saga.expected_origin,
                        "expected_owner_id": saga.expected_owner_id,
                        "expected_receiving_bot_ref": saga.expected_receiving_bot_ref,
                        "expected_root_ref": saga.expected_root_ref,
                        "displaced_legacy_operation_id": (
                            saga.displaced_legacy_operation_id
                        ),
                    }
                    for operation_id, saga in sorted(
                        self._managed_chat_binding_sagas.items()
                    )
                },
            }
            with self._file_lock(True):
                self._write_atomic(data)
            self._rebuild_bound_chat_index()
            return True
        except ProjectCommitUncertainError as exc:
            logger.error("保存项目数据耐久性不确定: %s", get_error_detail(exc))
            raise
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

    def quarantine_bound_chat(self, chat_id: str) -> bool:
        """Fail closed when compensating persistence cannot be confirmed."""

        if not chat_id:
            return False
        with self._lock:
            self._quarantined_bound_chat_ids.add(chat_id)
            self._bound_chat_index.pop(chat_id, None)
            return self._save_projects()

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
            quarantined = data.get("quarantined_bound_chat_ids", [])
            if not isinstance(quarantined, list) or any(
                not isinstance(chat_id, str) or not chat_id
                for chat_id in quarantined
            ):
                raise ValueError("invalid quarantined_bound_chat_ids")
            self._quarantined_bound_chat_ids = set(quarantined)
            residuals = data.get("managed_group_residuals", {})
            if not isinstance(residuals, dict):
                raise ValueError("invalid managed_group_residuals")
            parsed_residuals: dict[str, tuple[str, str]] = {}
            for operation_id, value in residuals.items():
                if (
                    not isinstance(operation_id, str)
                    or not operation_id
                    or not isinstance(value, dict)
                    or set(value) != {"chat_id", "delete_state"}
                    or not isinstance(value["chat_id"], str)
                    or not value["chat_id"]
                    or value["delete_state"]
                    not in {
                        "create_outcome_unknown",
                        "delete_rejected",
                        "delete_unknown",
                        "delete_guarded",
                        "recovered_chat_invalid",
                        "registry_bind_uncertain",
                        "untrusted_retained",
                        "legacy_saga_resolution_required",
                    }
                ):
                    raise ValueError("invalid managed group residual")
                parsed_residuals[operation_id] = (
                    value["chat_id"],
                    value["delete_state"],
                )
            self._managed_group_residuals = parsed_residuals
            sagas = data.get("managed_chat_binding_sagas", {})
            if not isinstance(sagas, dict):
                raise ValueError("invalid managed_chat_binding_sagas")
            parsed_sagas: dict[str, ManagedChatBindingSaga] = {}
            for operation_id, value in sagas.items():
                if not isinstance(value, dict) or frozenset(value) not in {
                    frozenset({"project_id", "snapshot"}),
                    frozenset(
                        {
                            "chat_id",
                            "project_id",
                            "remove_project_on_restore",
                            "snapshot",
                        }
                    ),
                    frozenset(
                        {
                            "chat_id",
                            "project_id",
                            "remove_project_on_restore",
                            "snapshot",
                            "expected",
                            "expected_origin",
                            "expected_owner_id",
                            "expected_receiving_bot_ref",
                            "expected_root_ref",
                        }
                    ),
                    frozenset(
                        {
                            "chat_id",
                            "project_id",
                            "remove_project_on_restore",
                            "snapshot",
                            "expected",
                            "expected_origin",
                            "expected_owner_id",
                            "expected_receiving_bot_ref",
                            "expected_root_ref",
                            "displaced_legacy_operation_id",
                        }
                    ),
                }:
                    raise ValueError("invalid managed chat binding saga")
                snapshot = value["snapshot"]
                if not isinstance(snapshot, dict) or frozenset(snapshot) not in {
                    frozenset(
                        {
                            "allowed_chat_ids",
                            "bound_chat_created_at",
                            "bound_chat_id",
                            "bound_chat_name",
                        }
                    ),
                    frozenset(
                        {
                            "allowed_chat_ids",
                            "bound_chat_created_at",
                            "bound_chat_id",
                            "bound_chat_name",
                            "owner_chat_id",
                            "project_name",
                        }
                    ),
                    frozenset(
                        {
                            "allowed_chat_ids",
                            "binding_generation",
                            "bound_chat_created_at",
                            "bound_chat_id",
                            "bound_chat_name",
                            "owner_chat_id",
                            "project_name",
                        }
                    ),
                }:
                    raise ValueError("invalid managed chat binding saga snapshot")
                project_id = value["project_id"]
                project = self._projects.get(project_id)
                chat_id = value.get("chat_id") or (
                    project.bound_chat_id if project is not None else ""
                )
                if (
                    not isinstance(operation_id, str)
                    or not operation_id
                    or not isinstance(project_id, str)
                    or not project_id
                    or not isinstance(chat_id, str)
                    or not chat_id
                    or type(value.get("remove_project_on_restore", False)) is not bool
                    or not isinstance(
                        value.get("displaced_legacy_operation_id", ""), str
                    )
                ):
                    raise ValueError("invalid managed chat binding saga identity")
                parsed_snapshot = ManagedChatBindingSnapshot(
                    bound_chat_id=snapshot["bound_chat_id"],
                    bound_chat_name=snapshot["bound_chat_name"],
                    bound_chat_created_at=float(snapshot["bound_chat_created_at"]),
                    allowed_chat_ids=tuple(
                        (str(item[0]), float(item[1]))
                        for item in snapshot["allowed_chat_ids"]
                    ),
                    owner_chat_id=snapshot.get(
                        "owner_chat_id",
                        project.owner_chat_id if project is not None else "",
                    ),
                    project_name=snapshot.get(
                        "project_name",
                        project.project_name if project is not None else "",
                    ),
                    binding_generation=int(snapshot.get("binding_generation", 0)),
                )
                expected_raw = value.get("expected")
                if expected_raw is None:
                    expected_snapshot = None
                else:
                    if not isinstance(expected_raw, dict) or frozenset(
                        expected_raw
                    ) != frozenset(
                        {
                            "allowed_chat_ids",
                            "binding_generation",
                            "bound_chat_created_at",
                            "bound_chat_id",
                            "bound_chat_name",
                            "owner_chat_id",
                            "project_name",
                        }
                    ):
                        raise ValueError("invalid managed chat binding saga expected")
                    expected_snapshot = ManagedChatBindingSnapshot(
                        bound_chat_id=expected_raw["bound_chat_id"],
                        bound_chat_name=expected_raw["bound_chat_name"],
                        bound_chat_created_at=float(
                            expected_raw["bound_chat_created_at"]
                        ),
                        allowed_chat_ids=tuple(
                            (str(item[0]), float(item[1]))
                            for item in expected_raw["allowed_chat_ids"]
                        ),
                        owner_chat_id=expected_raw["owner_chat_id"],
                        project_name=expected_raw["project_name"],
                        binding_generation=int(
                            expected_raw["binding_generation"]
                        ),
                    )
                parsed_sagas[operation_id] = ManagedChatBindingSaga(
                    operation_id=operation_id,
                    project_id=project_id,
                    chat_id=chat_id,
                    snapshot=parsed_snapshot,
                    expected=expected_snapshot,
                    expected_origin=str(value.get("expected_origin") or ""),
                    expected_owner_id=str(value.get("expected_owner_id") or ""),
                    expected_receiving_bot_ref=str(
                        value.get("expected_receiving_bot_ref") or ""
                    ),
                    expected_root_ref=str(
                        value.get("expected_root_ref")
                        or (project.root_path if project is not None else "")
                    ),
                    remove_project_on_restore=value.get(
                        "remove_project_on_restore", False
                    ),
                    displaced_legacy_operation_id=str(
                        value.get("displaced_legacy_operation_id") or ""
                    ),
                )
            for saga in parsed_sagas.values():
                displaced_operation_id = saga.displaced_legacy_operation_id
                if not displaced_operation_id:
                    continue
                displaced = parsed_sagas.get(displaced_operation_id)
                if (
                    displaced is None
                    or displaced.expected is not None
                    or displaced.project_id != saga.project_id
                    or displaced.chat_id != saga.chat_id
                    or displaced.expected_root_ref != saga.expected_root_ref
                    or parsed_residuals.get(displaced_operation_id)
                    != (saga.chat_id, "legacy_saga_resolution_required")
                    or saga.chat_id not in self._quarantined_bound_chat_ids
                ):
                    raise ValueError("invalid displaced legacy binding saga")
            self._managed_chat_binding_sagas = parsed_sagas
            self._rebuild_bound_chat_index()
        except Exception as e:
            corrupt_path = Path(f"{self._storage_path}.corrupt.{int(time.time())}")
            try:
                if self._storage_path.exists():
                    os.replace(self._storage_path, corrupt_path)
                    logger.error("加载项目数据失败，已备份损坏文件到: %s", corrupt_path)
            except Exception:
                logger.error("加载项目数据失败: %s", get_error_detail(e))
