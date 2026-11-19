from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from .backends import ALL_BACKEND_METAS, InteractionMode

PROGRAMMING_MODES: frozenset[InteractionMode] = frozenset(
    spec.interaction_mode for spec in ALL_BACKEND_METAS
)
PROGRAMMING_MODE_VALUES: frozenset[str] = frozenset(
    mode.value for mode in PROGRAMMING_MODES
)


_MODE_DISPLAY_NAMES: dict[InteractionMode, str] = {
    InteractionMode.SMART: "🧠 智能模式",
    **{
        spec.interaction_mode: spec.mode_display_name
        for spec in ALL_BACKEND_METAS
    },
    InteractionMode.SHELL: "💻 Shell 模式",
}


def _make_enter_mode_method(target_mode: InteractionMode):
    """Build a typed ``enter_<mode>_mode`` bound accessor for ModeManager."""

    def _enter(
        self: ModeManager,
        chat_id: str,
        auto: bool = False,
        project_id: Optional[str] = None,
    ) -> InteractionMode:
        return self.enter_programming_mode(chat_id, target_mode, auto=auto, project_id=project_id)

    _enter.__name__ = f"enter_{target_mode.value}_mode"
    _enter.__doc__ = f"进入 {target_mode.value} 编程模式。"
    return _enter


def _make_is_mode_method(target_mode: InteractionMode):
    """Build a typed ``is_<mode>_mode`` predicate accessor for ModeManager."""

    def _is(
        self: ModeManager,
        chat_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        return self.get_mode(chat_id, project_id) == target_mode

    _is.__name__ = f"is_{target_mode.value}_mode"
    _is.__doc__ = f"判断当前是否为 {target_mode.value} 模式。"
    return _is


def _register_programming_mode_accessors(cls: type[ModeManager]) -> type[ModeManager]:
    """Attach symmetric ``enter_<mode>_mode`` / ``is_<mode>_mode`` helpers from PROGRAMMING_MODES."""
    for mode in PROGRAMMING_MODES:
        enter_name = f"enter_{mode.value}_mode"
        is_name = f"is_{mode.value}_mode"
        if not hasattr(cls, enter_name):
            setattr(cls, enter_name, _make_enter_mode_method(mode))
        if not hasattr(cls, is_name):
            setattr(cls, is_name, _make_is_mode_method(mode))
    return cls


@dataclass
class ModeState:
    """某个 chat 或 project 的模式状态。"""

    mode: InteractionMode = InteractionMode.SMART
    auto_entered: bool = False


class ModeManager:
    """Manages interaction modes at both chat and project levels.

    Mode resolution order:
    1. If project_id is provided and has a mode set, use project mode
    2. Otherwise, fall back to chat-level mode
    3. Default is SMART mode

    Project-level modes are keyed by ``{chat_id}:{project_id}`` so that
    different chats operating on the same project maintain independent modes.
    """

    def __init__(self):
        self._chat_modes: dict[str, ModeState] = {}
        # Key format: "{chat_id}:{project_id}" — ensures cross-chat isolation.
        self._project_modes: dict[str, ModeState] = {}
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    @staticmethod
    def _project_key(chat_id: str, project_id: str) -> str:
        """Build the composite key for ``_project_modes``."""
        return f"{chat_id}:{project_id}"

    def get_mode(self, chat_id: str, project_id: Optional[str] = None) -> InteractionMode:
        """Get the current interaction mode.

        Args:
            chat_id: The chat identifier
            project_id: Optional project identifier for project-level mode lookup

        Returns:
            The current interaction mode (project mode > chat mode > SMART)
        """
        with self._lock:
            if project_id:
                key = self._project_key(chat_id, project_id)
                project_state = self._project_modes.get(key)
                if project_state:
                    return project_state.mode
            chat_state = self._chat_modes.get(chat_id)
            return chat_state.mode if chat_state else InteractionMode.SMART

    def set_mode(
        self,
        chat_id: str,
        mode: InteractionMode,
        auto_entered: bool = False,
        project_id: Optional[str] = None,
    ) -> InteractionMode:
        """设置交互模式（chat 级或 project 级）。

        设计说明：
        - `project_id` 非空：只影响该 chat+project 组合；不同 chat 对同一 project 互不干扰。
        - `project_id` 为空：设置 chat 级默认模式（兼容旧行为）。
        - `auto_entered=True`：标记为"自动进入"（例如回复链路自动切换），便于 UI/审计区分。

        Args:
            chat_id: The chat identifier
            mode: The new mode to set
            auto_entered: Whether the mode was auto-entered (e.g., by replying to a message)
            project_id: If provided, set mode at project level; otherwise at chat level

        Returns:
            The previous mode
        """
        with self._lock:
            if project_id:
                key = self._project_key(chat_id, project_id)
                old_state = self._project_modes.get(key, ModeState())
                old_mode = old_state.mode
                self._project_modes[key] = ModeState(mode=mode, auto_entered=auto_entered)
            else:
                old_state = self._chat_modes.get(chat_id, ModeState())
                old_mode = old_state.mode
                self._chat_modes[chat_id] = ModeState(mode=mode, auto_entered=auto_entered)
            return old_mode

    def clear_project_mode(self, chat_id: str, project_id: str) -> Optional[InteractionMode]:
        """Clear the mode for a specific chat+project combination.

        Args:
            chat_id: The chat identifier
            project_id: The project identifier

        Returns:
            The previous mode if it existed, None otherwise
        """
        with self._lock:
            key = self._project_key(chat_id, project_id)
            old_state = self._project_modes.pop(key, None)
            return old_state.mode if old_state else None

    def clear_modes_for_chat(self, chat_id: str) -> int:
        """Remove all project-mode entries for *chat_id*.

        Called when a chat_id is evicted from a project's allowed_chat_ids
        via LRU, so that stale ``{chat_id}:{project_id}`` keys do not
        accumulate in ``_project_modes`` indefinitely.

        Returns:
            Number of entries removed.
        """
        prefix = f"{chat_id}:"
        with self._lock:
            stale_keys = [k for k in self._project_modes if k.startswith(prefix)]
            for k in stale_keys:
                del self._project_modes[k]
            return len(stale_keys)

    def get_project_mode(self, chat_id: str, project_id: str) -> Optional[InteractionMode]:
        """Get the mode for a specific chat+project (without fallback).

        Args:
            chat_id: The chat identifier
            project_id: The project identifier

        Returns:
            The project's mode if set, None otherwise
        """
        with self._lock:
            key = self._project_key(chat_id, project_id)
            state = self._project_modes.get(key)
            return state.mode if state else None

    def enter_programming_mode(
        self,
        chat_id: str,
        mode: InteractionMode,
        auto: bool = False,
        project_id: Optional[str] = None,
    ) -> InteractionMode:
        """统一进入编程模式入口。"""
        if mode not in PROGRAMMING_MODES:
            raise ValueError(f"mode must be a programming mode, got: {mode}")
        return self.set_mode(chat_id, mode, auto_entered=auto, project_id=project_id)

    # Alias for callers preferring the shorter name
    enter_mode = enter_programming_mode

    def exit_to_smart(self, chat_id: str, project_id: Optional[str] = None) -> InteractionMode:
        """退出到 SMART 模式。"""
        return self.set_mode(chat_id, InteractionMode.SMART, auto_entered=False, project_id=project_id)

    def is_programming_mode(self, chat_id: str, project_id: Optional[str] = None) -> bool:
        """判断是否处于编程模式。"""
        mode = self.get_mode(chat_id, project_id)
        return mode in PROGRAMMING_MODES

    def get_mode_display_name(self, chat_id: str, project_id: Optional[str] = None) -> str:
        """返回当前模式的 UI 展示名。"""
        mode = self.get_mode(chat_id, project_id)
        return _MODE_DISPLAY_NAMES.get(mode, "未知模式")


_register_programming_mode_accessors(ModeManager)
