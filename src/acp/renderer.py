"""ACP Event Renderer — converts ACP events to Feishu Markdown content.

Maintains state across events to build a complete view of:
- Agent text output (accumulated)
- Active tool calls (with kind-specific icons)
- Execution plan progress (checklist)
"""

from __future__ import annotations

import logging
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

    @staticmethod
    def _is_main_source(event: ACPEvent) -> bool:
        raw = getattr(event, "source_id", None)
        if raw is None:
            return True
        if isinstance(raw, str):
            return raw.strip() in ("", "main")
        return False

    def _ingest_event(self, event: ACPEvent) -> None:
        """Pure state mutation — update internal state from an ACP event."""
        is_main = self._is_main_source(event)
        match event.event_type:
            case ACPEventType.TEXT_CHUNK:
                if event.text and is_main:
                    self._text_chunks.append(event.text)
                    self._text_content += event.text

            case ACPEventType.THOUGHT_CHUNK:
                if event.text and is_main:
                    self._thought_chunks.append(event.text)

            case ACPEventType.TOOL_CALL_START:
                if event.tool_call:
                    self._active_tools[event.tool_call.id] = event.tool_call
                    for loc in event.tool_call.locations:
                        self._modified_files.add(loc)
                    # Track TodoWrite content
                    if event.tool_call.content:
                        self._todo_content = event.tool_call.content

            case ACPEventType.TOOL_CALL_UPDATE:
                if event.tool_call:
                    self._active_tools[event.tool_call.id] = event.tool_call
                    for loc in event.tool_call.locations:
                        self._modified_files.add(loc)
                    if event.tool_call.content:
                        self._todo_content = event.tool_call.content

            case ACPEventType.TOOL_CALL_DONE:
                if event.tool_call:
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
