from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EngineCardState:
    title: str = ""
    subtitle: str = ""
    content: str = ""
    progress_bar: Optional[str] = None
    project_id: Optional[str] = None
    engine_project_id: Optional[str] = None
    is_executing: bool = False
    is_paused: bool = False
    engine_name: str = "Coco"
    show_buttons: bool = True
    working_dir: Optional[str] = None
    status_line: Optional[str] = None
    duration_line: Optional[str] = None
    criteria_section: Optional[str] = None
    footer_note: Optional[str] = None
    compact: bool = False
    expanded: bool = False
    expand_ac: bool = False
    action_prefix: str = "deep"
    extra_buttons: Optional[list[dict]] = None
    warning_banner: Optional[str] = None
    # ── 消息卡片优化新增字段 ──
    terminal_state: Optional[str] = None  # running/completed/failed/cancelled/blocked/denied/continued
    is_read: bool = True  # 未读标记 (False → 标题前缀 🔴)
    footer_status: Optional[str] = None  # thinking/tool_running

    @property
    def deep_project_id(self):
        return self.engine_project_id


@dataclass
class EngineStatusEntry:
    """Aggregated engine status entry for unified diagnostics."""
    mode: str
    task_id: str
    name: str
    status: str
    info: str
    started_at: Optional[float] = None


DeepCardState = EngineCardState
