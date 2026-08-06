from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class ToolOptionView:
    """卡片层通用的“工具选择”选项模型。

    说明：
    - 与具体后端实现解耦，只承载 UI 所需字段；
    - name: 逻辑工具名称（如 coco/claude 等）；
    - description: 简要说明文案；
    - is_default: 是否作为默认高亮/推荐选项；
    - emoji: 可选图标，用于提高可读性；
    - disabled: 卡片上是否禁用该选项（例如工具不可用时）。
    """

    name: str
    description: str = ""
    is_default: bool = False
    emoji: str = "🤖"
    disabled: bool = False


@dataclass
class ModelOptionView:
    """卡片层通用的“模型选择”选项模型。"""

    name: str
    description: str = ""
    is_default: bool = False
    display_name: Optional[str] = None


@dataclass
class ReasoningState:
    """Reasoning block state for card rendering."""
    content: str = ""
    active: bool = False
    expanded: bool = False


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
    # 结构化内容（Phase 2: 引擎折叠面板支持）
    # 当 rendered_content 存在时，DeepBuilder 会使用 to_elements(collapsible=True) 替代纯 markdown
    rendered_content: Optional[object] = None  # RenderedContent from acp.renderer
    # ── 消息卡片优化新增字段 ──
    terminal_state: Optional[str] = None  # running/completed/failed/cancelled/blocked/awaiting_approval/denied/continued
    is_read: bool = True  # 未读标记 (False → 标题前缀 🔴)
    footer_status: Optional[str] = None  # thinking/tool_running/waiting_approval
    reasoning: Optional[ReasoningState] = None  # Reasoning block state

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


@dataclass
class CardLayoutSpec:
    """统一卡片布局规格 — 所有编程模式和引擎模式共享的布局参数。

    该 dataclass 是卡片系统的通用 layout 输入，
    UnifiedCardLayout.build() 根据字段是否存在自动选择输出对应 element。
    """

    # ---- 通用字段 ----
    project_path: Optional[str] = None
    image_keys: Optional[list[str]] = None
    buttons: Optional[list[dict]] = None

    # ---- 状态栏（流式卡片风格） ----
    status_color: str = "blue"
    error_count: int = 0
    progress_text: str = ""
    sticky_message: Optional[str] = None

    # ---- 引擎特有区域 ----
    progress_bar: Optional[str] = None
    status_line: Optional[str] = None
    duration_line: Optional[str] = None
    criteria_section: Optional[str] = None
    warning_banner: Optional[str] = None
    footer_note: Optional[str] = None
    engine_meta_separator: str = " · "

    # ---- 内容（二选一） ----
    content_markdown: Optional[str] = None
    content_elements: Optional[list[dict]] = None  # 预构建 Feishu elements (collapsible panels 等)

    # ---- 渲染选项 ----
    legacy_safe: bool = False  # PATCH 更新时需要 legacy-safe 元素（不含 text_size/element_id）
    content_element_id: str = "content_md"  # 非 legacy 模式的 element_id

    # ---- 预构建按钮 elements（引擎模式用，跳过 build_responsive_layout） ----
    button_elements: Optional[list[dict]] = None

    # ---- 消息卡片优化新增字段 ----
    footer_status: Optional[str] = None  # thinking/tool_running/waiting_approval
    terminal_state: Optional[str] = None  # completed/failed/cancelled/blocked etc.


class KeyEventKind(str, Enum):
    """卡片侧"关键事件"最小集合（跨 Normal/Deep/Spec 统一语义）。

    说明：这里的 kind 只描述"是什么类型的信息"，不绑定具体引擎实现。
    """

    STAGE = "stage"  # Deep: 阶段；Spec: phase
    TURN = "turn"  # Normal: 轮次
    PLAN = "plan"  # 计划/Checklist 更新
    TOOL = "tool"  # 工具调用（开始/进度/完成）
    ARTIFACT = "artifact"  # 产物（文件、链接、构建产物、报告等）
    ISSUE = "issue"  # 错误/告警/风险
    USER_ACTION = "user_action"  # 用户交互点（按钮点击、继续/停止等）
    CONCLUSION = "conclusion"  # 最终结论/交付摘要


class KeyEventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueDisposition(str, Enum):
    """错误/告警条目的处置状态（用于“错误与告警”面板）。"""

    UNHANDLED = "unhandled"  # 未处理
    IN_PROGRESS = "in_progress"  # 处理中
    RECOVERED = "recovered"  # 已恢复
    NEED_USER = "need_user"  # 需用户介入


@dataclass(frozen=True)
class KeyEvent:
    """用于卡片展示与交互定位的“关键事件”统一模型。

    字段契约（核心）：
    - event_id: 稳定标识（用于引用/定位/交互 value 中传递）
    - ts: 时间戳（秒，float）
    - kind: 事件类型（KeyEventKind）
    - title/body: 可展示文本（建议 Markdown）
    - refs: 被引用的事件ID列表（用于跨区域避免重复正文）
    - severity/disposition: ISSUE 事件的严重级别与处置状态
    - axis_key: 主轴定位键（如 turn:3 / iteration:2 / phase:Plan），帮助在卡片里快速定位
    """

    event_id: str
    ts: float
    kind: KeyEventKind
    title: str
    body: str = ""
    refs: tuple[str, ...] = ()
    severity: KeyEventSeverity = KeyEventSeverity.INFO
    disposition: Optional[IssueDisposition] = None
    axis_key: Optional[str] = None
    mode: Optional[str] = None  # Normal/Deep/Spec
    meta: dict = field(default_factory=dict)


def build_sample_key_events(now: Optional[float] = None) -> list[KeyEvent]:
    """生成一组可复用的样例关键事件（用于测试/回归/演示）。"""

    base = now if now is not None else time.time()
    return [
        KeyEvent(
            event_id="run:start",
            ts=base,
            kind=KeyEventKind.STAGE,
            title="执行开始",
            body="模式=Deep · engine=Coco",
            axis_key="stage:run",
            mode="Deep",
        ),
        KeyEvent(
            event_id="plan:1",
            ts=base + 1,
            kind=KeyEventKind.PLAN,
            title="计划更新",
            body="- ⏳ 读取代码\n- ⏳ 修改卡片渲染\n- ⏳ 运行测试",
            axis_key="stage:plan",
            mode="Deep",
        ),
        KeyEvent(
            event_id="tool:read:1",
            ts=base + 2,
            kind=KeyEventKind.TOOL,
            title="读取文件 src/acp/models.py",
            body="读取并解析 ACPEvent 定义",
            axis_key="stage:task",
            mode="Deep",
        ),
        KeyEvent(
            event_id="issue:1",
            ts=base + 3,
            kind=KeyEventKind.ISSUE,
            title="发现卡片存在折叠区块",
            body="需要移除 expand/collapse 交互，保证无折叠/省略。",
            severity=KeyEventSeverity.WARNING,
            disposition=IssueDisposition.IN_PROGRESS,
            refs=("tool:read:1",),
            axis_key="stage:analysis",
            mode="Deep",
        ),
        KeyEvent(
            event_id="run:done",
            ts=base + 10,
            kind=KeyEventKind.CONCLUSION,
            title="执行完成",
            body="状态=成功 · 产物=卡片渲染契约与测试通过",
            refs=("plan:1", "issue:1"),
            axis_key="stage:finish",
            mode="Deep",
        ),
    ]
