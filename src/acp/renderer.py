"""ACP Event Renderer — converts ACP events to Feishu Markdown content.

Maintains state across events to build a complete view of:
- Agent text output (accumulated)
- Active tool calls (with kind-specific icons)
- Execution plan progress (checklist)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Optional

from .models import ACPEvent, ACPEventType, PlanInfo, ToolCallInfo

logger = logging.getLogger(__name__)

# Tool kind → display icon
_KIND_ICONS = {
    "read": "📖",
    "edit": "✏️",
    "delete": "🗑️",
    "move": "📁",
    "search": "🔍",
    "execute": "⚡",
    "think": "🧠",
    "fetch": "🌐",
    "switch_mode": "🔄",
    "other": "🔧",
}

# Tool status → display icon
_STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "failed": "❌",
}


# ------------------------------------------------------------------
# Structured content model (for collapsible-panel rendering)
# ------------------------------------------------------------------
@dataclass
class ContentSection:
    """A logical section of rendered content."""

    section_type: str  # "text" | "tool_group" | "thought" | "plan" | "todo" | "active_tools"
    markdown: str
    tool_kind: str = ""
    tool_count: int = 0
    is_complete: bool = True
    collapsed_by_default: bool = False
    has_failure: bool = False


@dataclass
class RenderedContent:
    """Structured rendering result — can be serialised to flat markdown or to Feishu card elements."""

    sections: list[ContentSection] = field(default_factory=list)

    # ---- backward-compat helpers ----

    def to_elements(self, *, collapsible: bool = True) -> list[dict]:
        """Convert sections to Feishu card Schema 2.0 elements.

        When *collapsible* is True, ``tool_group`` and ``thought`` sections are
        wrapped in ``collapsible_panel``; otherwise everything becomes plain
        ``markdown`` elements.

        Tool grouping strategy (collapsible mode only):
        - ≤2 consecutive completed tool calls → individual panels
        - >2 consecutive completed tool calls → merged into one "☕ N个工具调用" panel
        """
        if not collapsible:
            return [{"tag": "markdown", "content": sec.markdown}
                    for sec in self.sections if sec.markdown]

        elements: list[dict] = []
        i = 0
        while i < len(self.sections):
            sec = self.sections[i]
            if not sec.markdown:
                i += 1
                continue

            if sec.collapsed_by_default and sec.section_type == "tool_group" and sec.is_complete:
                # Collect consecutive completed tool_group sections
                group_start = i
                merged_md_parts: list[str] = []
                total_tools = 0
                any_failure = False
                while i < len(self.sections):
                    s = self.sections[i]
                    if s.section_type == "tool_group" and s.is_complete and s.collapsed_by_default and s.markdown:
                        merged_md_parts.append(s.markdown)
                        total_tools += max(s.tool_count, 1)
                        if s.has_failure:
                            any_failure = True
                        i += 1
                    else:
                        break

                group_count = i - group_start
                if group_count == 1 or total_tools <= 2:
                    # Individual panels
                    for j in range(group_start, i):
                        s = self.sections[j]
                        if s.markdown:
                            elements.append(self._wrap_collapsible(s))
                else:
                    # Merge into grouped panel
                    merged_sec = ContentSection(
                        section_type="tool_group",
                        markdown="\n".join(merged_md_parts),
                        tool_kind="",
                        tool_count=total_tools,
                        is_complete=True,
                        collapsed_by_default=True,
                        has_failure=any_failure,
                    )
                    elements.append(self._wrap_grouped_tools(merged_sec))
            elif sec.collapsed_by_default and sec.section_type in ("tool_group", "thought"):
                elements.append(self._wrap_collapsible(sec))
                i += 1
            else:
                elements.append({"tag": "markdown", "content": sec.markdown})
                i += 1
        return elements

    @staticmethod
    def _wrap_grouped_tools(sec: ContentSection) -> dict:
        """Wrap merged tool sections into a single grouped collapsible panel."""
        from src.card.themes import PANEL_STYLES

        header_text = f"☕ **{sec.tool_count}个工具调用**（已结束）"
        border_color = PANEL_STYLES["border_failed"] if sec.has_failure else PANEL_STYLES["border_history"]

        return {
            "tag": "collapsible_panel",
            "expanded": False,
            "header": {"tag": "markdown", "content": header_text},
            "border": {"color": border_color},
            "background_color": "bg-fill-tag-neutral",
            "vertical_spacing": PANEL_STYLES["vertical_spacing"],
            "padding": PANEL_STYLES["padding"],
            "corner_radius": PANEL_STYLES["corner_radius"],
            "elements": [{"tag": "markdown", "content": sec.markdown}],
        }

    @staticmethod
    def _wrap_collapsible(sec: ContentSection) -> dict:
        """Wrap a section in a ``collapsible_panel`` element."""
        from src.card.themes import PANEL_STYLES

        if sec.section_type == "tool_group":
            icon = _KIND_ICONS.get(sec.tool_kind, "🔧")
            header_text = f"{icon} {sec.tool_count} 次工具调用" if sec.tool_count > 1 else f"{icon} 工具调用"
        elif sec.section_type == "thought":
            if not sec.is_complete:
                header_text = "🧠 正在分析"
            else:
                header_text = "🧠 过程摘要，点击查看"
        else:
            header_text = "详情"

        # Dynamic border color based on section state
        if sec.section_type == "tool_group" and sec.has_failure:
            border_color = PANEL_STYLES["border_failed"]
        elif sec.section_type == "tool_group" and sec.is_complete:
            border_color = PANEL_STYLES["border_history"]
        else:
            border_color = PANEL_STYLES["border_normal"]

        md_element: dict = {"tag": "markdown", "content": sec.markdown}
        if sec.section_type == "thought":
            md_element["text_size"] = "normal"

        return {
            "tag": "collapsible_panel",
            "expanded": False,
            "header": {"tag": "markdown", "content": f"**{header_text}**"},
            "border": {"color": border_color},
            "background_color": "bg-fill-tag-neutral",
            "vertical_spacing": PANEL_STYLES["vertical_spacing"],
            "padding": PANEL_STYLES["padding"],
            "corner_radius": PANEL_STYLES["corner_radius"],
            "elements": [md_element],
        }


@dataclass
class _MutableTurn:
    """Internal mutable turn representation."""

    reasoning_chunks: list[str] = field(default_factory=list)
    tools: dict[str, ToolCallInfo] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnSnapshot:
    """Read-only ACP turn snapshot for card v2 rendering."""

    reasoning: str = ""
    tools: tuple[ToolCallInfo, ...] = ()


class ACPEventRenderer:
    """Converts ACP events into Feishu-displayable Markdown content."""

    def __init__(self):
        self._text_chunks: list[str] = []
        self._text_content: str = ""
        self._text_dirty: bool = False
        self._active_tools: dict[str, ToolCallInfo] = {}
        self._completed_tool_count: int = 0
        self._plan: Optional[PlanInfo] = None
        self._modified_files: set[str] = set()
        self._todo_content: str = ""  # Latest TodoWrite rendered content
        self._thought_chunks: list[str] = []  # Accumulated thought text for structured rendering
        self._turns: list[_MutableTurn] = []

    def _record_text_turn(self, text: str) -> None:
        if not text:
            return
        if not self._turns or self._turns[-1].tools:
            self._turns.append(_MutableTurn())
        self._turns[-1].reasoning_chunks.append(text)

    def _record_tool_turn(self, tool_call: ToolCallInfo) -> None:
        if not tool_call:
            return
        for turn in reversed(self._turns):
            if tool_call.id in turn.tools:
                turn.tools[tool_call.id] = tool_call
                return
        if not self._turns:
            self._turns.append(_MutableTurn())
        self._turns[-1].tools[tool_call.id] = tool_call

    def snapshot_turns(self) -> tuple[TurnSnapshot, ...]:
        """Return immutable reasoning/tool turns without changing legacy markdown."""
        snapshots: list[TurnSnapshot] = []
        for turn in self._turns:
            reasoning = "".join(turn.reasoning_chunks)
            tools = tuple(replace(tool) for tool in turn.tools.values())
            if reasoning or tools:
                snapshots.append(TurnSnapshot(reasoning=reasoning, tools=tools))
        return tuple(snapshots)

    def _ingest_event(self, event: ACPEvent) -> None:
        """Pure state mutation — update internal state from an ACP event."""
        match event.event_type:
            case ACPEventType.TEXT_CHUNK:
                if event.text:
                    self._record_text_turn(event.text)
                    self._text_chunks.append(event.text)
                    self._text_content += event.text

            case ACPEventType.THOUGHT_CHUNK:
                if event.text:
                    self._thought_chunks.append(event.text)

            case ACPEventType.TOOL_CALL_START:
                if event.tool_call:
                    self._record_tool_turn(event.tool_call)
                    self._active_tools[event.tool_call.id] = event.tool_call
                    for loc in event.tool_call.locations:
                        self._modified_files.add(loc)
                    # Track TodoWrite content
                    if event.tool_call.content:
                        self._todo_content = event.tool_call.content

            case ACPEventType.TOOL_CALL_UPDATE:
                if event.tool_call:
                    self._record_tool_turn(event.tool_call)
                    self._active_tools[event.tool_call.id] = event.tool_call
                    for loc in event.tool_call.locations:
                        self._modified_files.add(loc)
                    if event.tool_call.content:
                        self._todo_content = event.tool_call.content

            case ACPEventType.TOOL_CALL_DONE:
                if event.tool_call:
                    self._record_tool_turn(event.tool_call)
                    tool_id = event.tool_call.id
                    self._completed_tool_count += 1
                    self._active_tools.pop(tool_id, None)
                    for loc in event.tool_call.locations:
                        self._modified_files.add(loc)

                    if event.tool_call.content:
                        # TodoWrite: update dedicated section, don't pollute text buffer
                        self._todo_content = event.tool_call.content
                    # Tool completion rendering is handled by activity_digest
                    # in the card render layer (flatten_to_atoms), no longer
                    # injected as inline text here.

            case ACPEventType.PLAN_UPDATE:
                if event.plan:
                    self._plan = event.plan

    def process_event(self, event: ACPEvent) -> str:
        """Process an event and return the current complete rendered content."""
        self._ingest_event(event)
        return self._render()

    def get_final_content(self) -> str:
        """Return the final rendered content (no active tools shown).

        If no text output was produced but thought chunks exist, includes
        thought content as fallback so the user sees *something* meaningful
        rather than a blank "执行完成" message.
        """
        self._active_tools = {}
        rendered = self._render()
        if not rendered and self._thought_chunks:
            thought_text = "".join(self._thought_chunks).strip()
            if thought_text:
                rendered = f"🧠 **思考过程**\n{thought_text}"
        return rendered

    @property
    def text_content(self) -> str:
        """Raw accumulated text."""
        if self._text_dirty:
            self._text_content = "".join(self._text_chunks)
            self._text_dirty = False
        return self._text_content

    @property
    def modified_files(self) -> set[str]:
        """Set of file paths modified during this session (read-only view)."""
        return self._modified_files

    @property
    def todo_content(self) -> str:
        """Latest TodoWrite content."""
        return self._todo_content

    @property
    def completed_tool_count(self) -> int:
        return self._completed_tool_count

    def reset(self) -> None:
        """Reset all accumulated state for a fresh rendering cycle.

        Call this method when reusing the same ``ACPEventRenderer`` instance for
        a new round of ACP prompt processing — for example, before issuing a
        retry in ``DeepEngine`` or starting a new user prompt within the same
        session.  After calling ``reset()``, the renderer behaves identically to
        a newly constructed instance.

        **What gets reset**:

        - Accumulated agent text output
        - Tool call records (active calls and completed count)
        - Execution plan snapshot
        - File change tracking and todo content

        .. note::
           For the full list of internal fields cleared by this method,
           see ``__init__``.

        **Important notes**:

        - This method **only** resets the renderer's internal accumulation
          state.  It does **not** trigger any cleanup callbacks, cancel running
          ACP sessions, or notify external subscribers.
        - Any external references previously obtained via properties (e.g.
          ``text_content``, ``modified_files``) remain **independent** after
          ``reset()`` — they still hold the data captured before the reset.
          Internally, ``reset()`` rebinds each mutable container to a fresh
          instance (matching ``__init__`` semantics), so prior references are
          never mutated.
        - If ``reset()`` is called while the renderer is in the middle of
          processing events (i.e. between ``process_event`` calls), subsequent
          ``process_event`` calls will start from a clean state — prior text,
          tool calls, and plan information will be lost.

        Example::

            renderer = ACPEventRenderer()
            # … process events for prompt 1 …
            content = renderer.get_final_content()

            # Reuse for prompt 2
            renderer.reset()
            # … process events for prompt 2 …
        """
        self._text_chunks = []
        self._text_content = ""
        self._text_dirty = False
        self._active_tools = {}
        self._completed_tool_count = 0
        self._plan = None
        self._modified_files = set()
        self._todo_content = ""
        self._thought_chunks = []
        self._turns = []

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self) -> str:
        """Render complete content: plan + todo + active tools + text."""
        parts: list[str] = []

        if self._plan:
            rendered_plan = self._render_plan()
            if rendered_plan:
                parts.append(rendered_plan)

        if self._todo_content:
            parts.append(f"**📝 任务进度**\n{self._todo_content}")

        if self._active_tools:
            rendered_tools = self._render_active_tools()
            if rendered_tools:
                parts.append(rendered_tools)

        text_content = self.text_content
        if text_content:
            parts.append(text_content)

        return "\n".join(parts) if parts else ""

    def _render_plan(self) -> str:
        """Render plan as a checklist."""
        if not self._plan or not self._plan.entries:
            return ""

        lines: list[str] = []
        for entry in self._plan.entries:
            content = (entry.content or "").strip()
            if not content:
                continue
            icon = _STATUS_ICONS.get(entry.status, "⬜")
            lines.append(f"{icon} {content}")

        if not lines:
            return ""
        return "\n".join(["**📋 执行计划**", *lines])

    def _render_active_tools(self) -> str:
        """Render currently active tool calls, grouped by kind to reduce noise."""
        groups: dict[str, list[tuple[str, str]]] = {}
        order: list[str] = []
        for tool in self._active_tools.values():
            if tool.content:
                continue
            title = (tool.title or "").strip()
            if tool.status not in ("in_progress", "pending") or not title:
                continue
            loc = tool.locations[0] if tool.locations else ""
            kind = tool.kind or "other"
            if kind not in groups:
                groups[kind] = []
                order.append(kind)
            groups[kind].append((title, loc))

        lines: list[str] = []
        for kind in order:
            items = groups[kind]
            icon = _KIND_ICONS.get(kind, "🔧")
            if len(items) == 1:
                title, loc = items[0]
                loc_str = f" `{loc}`" if loc else ""
                lines.append(f"{icon} {title}{loc_str}...")
            else:
                locs = [location for _, location in items if location]
                shown = locs if locs else [t for t, _ in items]
                sample = ", ".join(f"`{s}`" if locs else s for s in shown[:3])
                more = f" (+{len(shown) - 3})" if len(shown) > 3 else ""
                lines.append(f"{icon} {len(items)} 个进行中: {sample}{more}...")
        return "\n".join(lines) if lines else ""

    def reset_for_continuation(self, summary: str = "") -> None:
        """Reset state for a continuation card, optionally seeding a summary prefix.

        Like ``reset()``, clears all accumulated state.  If *summary* is
        provided, it is injected as the first text chunk so that subsequent
        renders include a brief context section at the top of the new card.
        """
        self.reset()
        if summary:
            self._record_text_turn(summary)
            self._text_chunks.append(summary)
            self._text_content = summary
