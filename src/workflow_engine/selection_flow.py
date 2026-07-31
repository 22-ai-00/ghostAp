"""Selection flow controller for the two-step orchestrator + review agent selection.

Encapsulates the state machine and Feishu card construction used by the
``/wf`` entry path. The handler invokes methods on this class instead of
building selection cards inline so that selection logic stays pure (no
Feishu I/O) and can be unit tested independently.

The flow is intentionally simple:

1. Orchestrator step: the user picks one tool + model combination that will
   drive the top-level script generation.
2. Review step: the user picks additional tool + model combinations that act
   as independent reviewers *or* clicks the "Auto" shortcut to skip review
   entirely (meaning the orchestrator also self-reviews).

Selections are stored as dicts keyed by ``selection_key`` so the handler can
add / remove items without having to parse card state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from src.card.actions.dispatch import (
    WORKFLOW_CANCEL,
    WORKFLOW_ORCHESTRATOR_SELECT_MODEL,
    WORKFLOW_ORCHESTRATOR_SELECT_MODEL_EFFORT,
    WORKFLOW_ORCHESTRATOR_SELECT_MODEL_GROUP,
    WORKFLOW_ORCHESTRATOR_SELECT_MODEL_PROFILE,
    WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
    WORKFLOW_REVIEW_SELECT_MODEL,
    WORKFLOW_REVIEW_SELECT_MODEL_EFFORT,
    WORKFLOW_REVIEW_SELECT_MODEL_GROUP,
    WORKFLOW_REVIEW_SELECT_MODEL_PROFILE,
    WORKFLOW_REVIEW_SELECT_TOOL,
)
from src.card.builder import CardBuilder
from src.card.render import model_cascade
from src.card.render.buttons import build_responsive_button_row

_MODEL_BUTTON_LABEL_MAX_CHARS = 32
_SELECT_LABEL_MAX_CHARS = 72
# Kept as module aliases so existing references stay valid; the source of
# truth now lives in ``src.card.render.model_cascade``.
_VARIANT_TOKENS = model_cascade.VARIANT_TOKENS
_PROFILE_ORDER = model_cascade.PROFILE_ORDER
_EFFORT_ORDER = model_cascade.EFFORT_ORDER

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SelectionItem:
    """A single tool + model selection stored by the controller.

    Attributes mirror the button payload shape the handlers round-trip
    through Feishu cards so that ``value`` dicts produced by this module can
    be fed back into :meth:`SelectionFlowController.add_or_update_selection`.
    """

    selection_key: str
    tool_name: str
    provider: str = "workflow"
    display_name: str = ""
    supports_model: bool = True
    model_name: Optional[str] = None
    use_default_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "selection_key": self.selection_key,
            "tool_name": self.tool_name,
            "provider": self.provider,
            "display_name": self.display_name or self.tool_name,
            "supports_model": self.supports_model,
        }
        if self.use_default_model:
            result["use_default_model"] = True
            result["model_name"] = ""
        elif self.model_name:
            result["model_name"] = self.model_name
            result["name"] = self.model_name
        else:
            result["use_default_model"] = True
        return result

    def label(self) -> str:
        """Human readable label used in card UI."""
        name = self.display_name or self.tool_name
        if self.use_default_model or not self.model_name:
            return f"`{name}` (默认模型)"
        return f"`{name}` · {self.model_name}"


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SelectionFlowController:
    """Pure-logic controller for the orchestrator + review selection flow.

    The controller owns no I/O — it only tracks selection state and produces
    plain Python dicts describing Feishu cards. The surrounding handler is
    responsible for sending / updating cards over Feishu and for forwarding
    button payloads back to the controller.
    """

    def __init__(
        self,
        step: int = 1,
        pending_tool_name: Optional[str] = None,
    ) -> None:
        if step not in (1, 2, 3):
            raise ValueError(f"step must be 1, 2, or 3, got {step!r}")
        self.step = step
        self.pending_tool_name: Optional[str] = pending_tool_name
        # selection_key -> dict (kept as dicts for easy JSON round-tripping)
        self.orchestrator_selections: dict[str, dict[str, Any]] = {}
        self.review_selections: dict[str, dict[str, Any]] = {}
        self.review_auto_mode: bool = False
        self.model_page: int = 0
        self.pending_model_group: Optional[str] = None
        self.pending_model_profile: Optional[str] = None
        self.pending_model_effort: Optional[str] = None
        # Error message surfaced by the handler on empty-submit failures.
        # The handler sets it and the card builders render it if present.
        self.error_message: str = ""

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    def set_step(self, step: int) -> None:
        if step not in (1, 2, 3):
            raise ValueError(f"step must be 1, 2, or 3, got {step!r}")
        if step == self.step:
            return
        self.step = step
        self.pending_tool_name = None
        self.model_page = 0
        self._reset_model_cascade()

    def finish_step(self) -> tuple[int, dict[str, dict[str, Any]]]:
        """Return (next_step, snapshot_of_current_step_selections).

        Called by the handler when the user confirms the current step. The
        return value is intended for persistence (the handler stores the
        snapshot on the session / project context so it survives card
        refreshes).
        """
        if self.step == 1:
            snapshot = dict(self.orchestrator_selections)
            next_step = 2
        elif self.step == 2:
            snapshot = dict(self.review_selections)
            next_step = 3
        else:
            # Step 3 is confirmation, no selections to snapshot
            snapshot = {}
            next_step = 1  # loop back defensively
        return next_step, snapshot

    def is_complete(self) -> bool:
        """True once orchestrator is chosen and review is chosen OR Auto."""
        if self.step == 3:
            return True
        if not self.orchestrator_selections:
            return False
        return bool(self.review_selections) or self.review_auto_mode

    # ------------------------------------------------------------------
    # Tool panel expand / collapse
    # ------------------------------------------------------------------

    def toggle_tool_expand(self, tool_name: str, *, is_review: bool) -> None:
        """Toggle the inline model-panel expansion for ``tool_name``.

        Only one tool is expanded at a time per flow. Expanding a tool while
        another is already expanded collapses the previous one; expanding a
        tool that is already expanded collapses it (clears the pending
        tool name).
        """
        if is_review and tool_name:
            self.review_auto_mode = False
        if not tool_name:
            self.pending_tool_name = None
            self.model_page = 0
            self._reset_model_cascade()
            return
        if self.pending_tool_name == tool_name:
            self.pending_tool_name = None
            self.model_page = 0
            self._reset_model_cascade()
        else:
            self.pending_tool_name = tool_name
            self.model_page = 0
            self._reset_model_cascade()

    def select_tool(self, tool_name: str, *, is_review: bool) -> None:
        """Select ``tool_name`` from a dropdown and keep its model panel open."""
        self.pending_tool_name = str(tool_name or "").strip() or None
        if is_review and self.pending_tool_name:
            self.review_auto_mode = False
        self.model_page = 0
        self._reset_model_cascade()

    def set_model_page(self, tool_name: str, page: int, *, is_review: bool) -> None:
        """Keep ``tool_name`` expanded and move its inline model panel page."""
        if is_review and tool_name:
            self.review_auto_mode = False
        if not tool_name:
            self.pending_tool_name = None
            self.model_page = 0
            self._reset_model_cascade()
            return
        self.pending_tool_name = tool_name
        self.model_page = max(0, int(page))

    def set_model_group(self, tool_name: str, model_group: str, *, is_review: bool) -> None:
        """Select the base model/family in the cascading model picker."""
        if is_review and tool_name:
            self.review_auto_mode = False
        self.pending_tool_name = str(tool_name or "").strip() or self.pending_tool_name
        self.pending_model_group = str(model_group or "").strip() or None
        self.pending_model_profile = None
        self.pending_model_effort = None
        self.model_page = 0

    def set_model_profile(self, tool_name: str, model_profile: str, *, is_review: bool) -> None:
        """Select the profile dimension for the current model family."""
        if is_review and tool_name:
            self.review_auto_mode = False
        self.pending_tool_name = str(tool_name or "").strip() or self.pending_tool_name
        self.pending_model_profile = str(model_profile or "").strip() or None
        self.pending_model_effort = None
        self.model_page = 0

    def set_model_effort(self, tool_name: str, model_effort: str, *, is_review: bool) -> None:
        """Select the effort dimension for the current model family/profile."""
        if is_review and tool_name:
            self.review_auto_mode = False
        self.pending_tool_name = str(tool_name or "").strip() or self.pending_tool_name
        self.pending_model_effort = str(model_effort or "").strip() or None
        self.model_page = 0

    # ------------------------------------------------------------------
    # Selection mutation
    # ------------------------------------------------------------------

    def _selection_store(self, *, is_review: bool) -> dict[str, dict[str, Any]]:
        return self.review_selections if is_review else self.orchestrator_selections

    def add_or_update_selection(
        self,
        selection: dict[str, Any],
        *,
        is_review: bool,
        keep_panel_open: bool = False,
    ) -> str:
        """Insert or update a selection based on its ``selection_key``.

        If no key is provided one is generated. Returns the key used for
        storage. When ``keep_panel_open`` is True (used by multi-select review
        step), the model panel stays expanded so the user can quickly add
        another model from the same tool.
        """
        key = str(selection.get("selection_key") or uuid.uuid4().hex)
        normalized = dict(selection)
        normalized["selection_key"] = key
        # Ensure a display name exists for rendering
        if not normalized.get("display_name"):
            normalized["display_name"] = normalized.get("tool_name", "")

        # Dedup: reject if exact tool+model combo already exists
        store = self._selection_store(is_review=is_review)
        if is_review:
            self.review_auto_mode = False
            tool = normalized.get("tool_name", "")
            model = normalized.get("model_name", "")
            use_default = normalized.get("use_default_model", False)
            for existing in store.values():
                if (
                    existing.get("tool_name") == tool
                    and existing.get("model_name") == model
                    and existing.get("use_default_model", False) == use_default
                ):
                    # Exact tool+model duplicate — skip silently
                    return existing["selection_key"]

        store[key] = normalized
        if keep_panel_open:
            # Keep panel open for rapid multi-select from same tool
            pass
        else:
            self.pending_tool_name = None
            self.model_page = 0
        return key

    def remove_selection(self, selection_key: str, *, is_review: bool) -> None:
        store = self._selection_store(is_review=is_review)
        store.pop(selection_key, None)

    def clear_selections(self, *, is_review: bool) -> None:
        self._selection_store(is_review=is_review).clear()

    def set_review_auto_mode(self, auto: bool) -> None:
        self.review_auto_mode = bool(auto)
        if auto:
            # Auto mode obviates the explicit review list; clear it so the
            # card no longer renders stale entries.
            self.review_selections.clear()
            self.pending_tool_name = None
            self.model_page = 0
            self._reset_model_cascade()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_non_empty(self, *, is_review: bool) -> tuple[bool, str]:
        """Return (ok, error_message) for the current step.

        Auto mode on the review step short-circuits — there is nothing to
        select. The orchestrator step always requires at least one selection.
        """
        if is_review and self.review_auto_mode:
            return True, ""
        store = self._selection_store(is_review=is_review)
        if not store:
            if is_review:
                return False, "请至少选择一个评审 Agent，或启用 Auto 跳过独立评审。"
            return False, "请至少选择一个主编排 Agent（工具 + 模型）。"
        return True, ""

    # ------------------------------------------------------------------
    # Snapshot helpers (used by handlers for persistence)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "pending_tool_name": self.pending_tool_name,
            "model_page": self.model_page,
            "pending_model_group": self.pending_model_group,
            "pending_model_profile": self.pending_model_profile,
            "pending_model_effort": self.pending_model_effort,
            "orchestrator_selections": dict(self.orchestrator_selections),
            "review_selections": dict(self.review_selections),
            "review_auto_mode": self.review_auto_mode,
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.step = int(data.get("step", 1))
        self.pending_tool_name = data.get("pending_tool_name")
        self.model_page = max(0, int(data.get("model_page", 0) or 0))
        self.pending_model_group = data.get("pending_model_group")
        self.pending_model_profile = data.get("pending_model_profile")
        self.pending_model_effort = data.get("pending_model_effort")
        self.orchestrator_selections = dict(data.get("orchestrator_selections", {}))
        self.review_selections = dict(data.get("review_selections", {}))
        self.review_auto_mode = bool(data.get("review_auto_mode", False))
        if self.review_auto_mode and self.review_selections:
            # Legacy snapshots could contain both after an explicit selection
            # was added from an Auto card. Preserve that latest explicit choice.
            self.review_auto_mode = False
        elif self.review_auto_mode:
            self.pending_tool_name = None
            self.model_page = 0
            self._reset_model_cascade()

    # ------------------------------------------------------------------
    # Card construction — orchestrator step
    # ------------------------------------------------------------------

    def build_orchestrator_combined_card(
        self,
        available_tools: list[dict[str, Any]],
        available_models: list[dict[str, Any]] | None = None,
        *,
        requirement: str = "",
        session_key: str = "",
        chat_id: str = "",
        project_id: str = "",
    ) -> dict[str, Any]:
        """Build the orchestrator-selection Feishu card.

        ``available_tools`` entries are expected to contain at least
        ``tool_name`` and optionally ``description`` / ``display_name``.
        ``available_models`` is optional — if provided, the inline model
        panel lists specific models; otherwise only the "默认模型" button is
        rendered.
        """
        return self._build_common_body(
            title="Workflow — 选择主编排 Agent",
            step_label="Step 1 / 3 — 主编排 Agent",
            selected_items=list(self.orchestrator_selections.values()),
            available_tools=available_tools,
            available_models=available_models or [],
            requirement=requirement,
            session_key=session_key,
            is_review=False,
            has_auto_shortcut=False,
            chat_id=chat_id,
            project_id=project_id,
        )

    def build_review_combined_card(
        self,
        available_tools: list[dict[str, Any]],
        available_models: list[dict[str, Any]] | None = None,
        *,
        requirement: str = "",
        session_key: str = "",
        chat_id: str = "",
        project_id: str = "",
    ) -> dict[str, Any]:
        """Build the review-agent selection Feishu card.

        Mirrors the orchestrator card but also exposes the "Auto" shortcut
        which bypasses explicit review selection entirely.
        """
        return self._build_common_body(
            title="Workflow — 选择评审 Agent",
            step_label="Step 2 / 3 — 评审 Agent",
            selected_items=list(self.review_selections.values()),
            available_tools=available_tools,
            available_models=available_models or [],
            requirement=requirement,
            session_key=session_key,
            is_review=True,
            has_auto_shortcut=True,
            chat_id=chat_id,
            project_id=project_id,
        )

    # ------------------------------------------------------------------
    # Shared body builder
    # ------------------------------------------------------------------

    def _build_common_body(
        self,
        *,
        title: str,
        step_label: str,
        selected_items: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        available_models: list[dict[str, Any]],
        requirement: str,
        session_key: str,
        is_review: bool,
        has_auto_shortcut: bool,
        chat_id: str = "",
        project_id: str = "",
    ) -> dict[str, Any]:
        """Construct the shared selection-card layout used by both steps."""
        elements: list[dict[str, Any]] = []

        # Stepper
        if is_review:
            stepper_markdown = (
                "▶ 1. 主编排 Agent  ✅\n"
                "▶ 2. 评审 Agent  — **当前**\n"
                "○ 3. 自动生成并执行"
            )
        else:
            stepper_markdown = (
                "▶ 1. 主编排 Agent  — **当前**\n"
                "○ 2. 评审 Agent\n"
                "○ 3. 自动生成并执行"
            )
        elements.append({
            "tag": "markdown",
            "content": f"**{step_label}**\n{stepper_markdown}",
        })

        # Requirement
        if requirement:
            truncated = requirement.strip()
            if len(truncated) > 200:
                truncated = truncated[:200] + "…"
            elements.append({"tag": "markdown", "content": f"**需求**：{truncated}"})

        elements.append({"tag": "hr"})

        # Error message
        if self.error_message:
            elements.append({
                "tag": "markdown",
                "content": f"⚠️ {self.error_message}",
            })
            elements.append({"tag": "hr"})

        # Auto shortcut for review step
        if has_auto_shortcut:
            auto_btn_text = (
                "✅ Auto 模式已启用（跳过独立评审）"
                if self.review_auto_mode
                else "Auto — 跳过独立评审 / 沿用主 Agent 能力"
            )
            auto_value = self._make_button_value(
                action="workflow_review_toggle_auto",
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
            )
            elements.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": auto_btn_text},
                "type": "primary" if self.review_auto_mode else "default",
                "value": auto_value,
                "behaviors": [{"type": "callback", "value": auto_value}],
            })
            elements.append({"tag": "hr"})

        # Selected items block
        if selected_items:
            lines = []
            for sel in selected_items:
                label = self._selection_label(sel)
                lines.append(f"✓ {label}")
                remove_value = self._make_button_value(
                    action=(
                        "workflow_review_remove" if is_review else "workflow_orchestrator_remove"
                    ),
                    session_key=session_key,
                    is_review=is_review,
                    chat_id=chat_id,
                    project_id=project_id,
                    selection_key=sel.get("selection_key", ""),
                )
                elements.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": f"移除 {label}"},
                    "type": "default",
                    "value": remove_value,
                    "behaviors": [{"type": "callback", "value": remove_value}],
                })
            clear_value = self._make_button_value(
                action=(
                    "workflow_review_clear" if is_review else "workflow_orchestrator_clear"
                ),
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
            )
            elements.append({
                "tag": "markdown",
                "content": "**已选**：\n" + "\n".join(lines),
            })
            elements.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": "清空全部"},
                "type": "default",
                "value": clear_value,
                "behaviors": [{"type": "callback", "value": clear_value}],
            })
            elements.append({"tag": "hr"})
        else:
            if not (is_review and self.review_auto_mode):
                elements.append({
                    "tag": "markdown",
                    "content": (
                        "尚未选择。在下方点击工具以展开模型面板，或直接使用默认模型。"
                    ),
                })
                elements.append({"tag": "hr"})

        # Tool + model cascading selectors. Large providers such as Traex expose
        # many variants; render one tool dropdown plus adaptive model dimensions
        # instead of dozens of callback buttons.
        elements.append({"tag": "markdown", "content": "**选择工具与模型**"})
        tool_select_action = WORKFLOW_REVIEW_SELECT_TOOL if is_review else WORKFLOW_ORCHESTRATOR_SELECT_TOOL
        selected_tool = self._current_tool(available_tools)
        fallback_tool = selected_tool or (dict(available_tools[0]) if available_tools else None)
        selected_tool_name = str(selected_tool.get("tool_name", "")) if selected_tool else ""
        tool_options = [
            self._select_option(
                self._label_with_description(
                    tool.get("display_name") or tool.get("tool_name") or "",
                    tool.get("description") or "",
                ),
                str(tool.get("tool_name") or ""),
            )
            for tool in available_tools
            if str(tool.get("tool_name") or "")
        ]
        tool_value = self._make_button_value(
            action=tool_select_action,
            session_key=session_key,
            is_review=is_review,
            chat_id=chat_id,
            project_id=project_id,
            tool_name=str(fallback_tool.get("tool_name", "")) if fallback_tool else "",
            provider=str(fallback_tool.get("provider", "workflow")) if fallback_tool else "workflow",
            display_name=str(fallback_tool.get("display_name") or fallback_tool.get("tool_name") or "") if fallback_tool else "",
            supports_model=bool(fallback_tool.get("supports_model", True)) if fallback_tool else True,
        )
        elements.append(
            self._select_static(
                placeholder="选择工具",
                value=tool_value,
                options=tool_options,
                initial_option=selected_tool_name or None,
            )
        )

        if selected_tool:
            self._append_model_cascade(
                elements,
                tool=selected_tool,
                available_models=available_models,
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
            )
        else:
            elements.append({"tag": "markdown", "content": "_先选择工具，再选择模型。_"})

        elements.append({"tag": "hr"})

        # Footer action
        finish_action = (
            "workflow_review_finish" if is_review else "workflow_orchestrator_finish"
        )
        finish_value = self._make_button_value(
            action=finish_action,
            session_key=session_key,
            is_review=is_review,
            chat_id=chat_id,
            project_id=project_id,
        )
        ok, _err = self.validate_non_empty(is_review=is_review)
        footer_label = (
            "确认评审选择，自动执行 →" if is_review else "确认主 Agent，下一步 →"
        )

        # Cancel button
        cancel_value = {
            "action": WORKFLOW_CANCEL,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": session_key,
        }

        # Build action buttons row with responsive layout
        action_buttons = [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "取消"},
                "type": "default",
                "value": cancel_value,
                "behaviors": [{"type": "callback", "value": cancel_value}],
                "confirm": {
                    "title": {"tag": "plain_text", "content": "确认取消 Workflow？"},
                    "text": {"tag": "plain_text", "content": "取消后已选择的工具将被清空，如需继续请重新发起。"},
                },
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": footer_label},
                "type": "primary" if ok else "default",
                "value": finish_value,
                "behaviors": [{"type": "callback", "value": finish_value}],
            },
        ]
        elements.extend(build_responsive_button_row(action_buttons, mobile_force_vertical=True))

        return CardBuilder._wrap_card(title, "blue", elements)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_model_cascade(self) -> None:
        self.pending_model_group = None
        self.pending_model_profile = None
        self.pending_model_effort = None

    def _current_tool(self, available_tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not available_tools:
            return None
        if self.pending_tool_name:
            for tool in available_tools:
                if str(tool.get("tool_name") or "") == self.pending_tool_name:
                    return dict(tool)
        return None

    def _append_model_cascade(
        self,
        elements: list[dict[str, Any]],
        *,
        tool: dict[str, Any],
        available_models: list[dict[str, Any]],
        session_key: str,
        is_review: bool,
        chat_id: str,
        project_id: str,
    ) -> None:
        tool_name = str(tool.get("tool_name") or "")
        display_name = str(tool.get("display_name") or tool_name)
        provider = str(tool.get("provider", "workflow"))
        supports_model = bool(tool.get("supports_model", True))
        model_action = WORKFLOW_REVIEW_SELECT_MODEL if is_review else WORKFLOW_ORCHESTRATOR_SELECT_MODEL

        default_value = self._make_button_value(
            action=model_action,
            session_key=session_key,
            is_review=is_review,
            chat_id=chat_id,
            project_id=project_id,
            tool_name=tool_name,
            provider=provider,
            display_name=display_name,
            supports_model=supports_model,
            use_default_model=True,
        )
        elements.extend(
            build_responsive_button_row([
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "默认模型（推荐）"},
                    "type": "primary",
                    "value": default_value,
                    "behaviors": [{"type": "callback", "value": default_value}],
                }
            ])
        )

        groups = self._build_model_groups(available_models)
        if not groups:
            elements.append({"tag": "markdown", "content": "_未配置额外模型列表；请选择默认模型。_"})
            return

        default_group, default_profile, default_effort = (
            model_cascade.resolve_default_selection(groups, None)
        )
        group_keys = [g["key"] for g in groups]
        if self.pending_model_group in group_keys:
            selected_group_key = self.pending_model_group
        elif default_group in group_keys:
            selected_group_key = default_group
        else:
            selected_group_key = group_keys[0]
        selected_group = next(g for g in groups if g["key"] == selected_group_key)
        group_default_profile, group_default_effort = (
            model_cascade.resolve_group_default_selection(selected_group)
        )
        group_action = (
            WORKFLOW_REVIEW_SELECT_MODEL_GROUP
            if is_review
            else WORKFLOW_ORCHESTRATOR_SELECT_MODEL_GROUP
        )
        group_value = self._make_button_value(
            action=group_action,
            session_key=session_key,
            is_review=is_review,
            chat_id=chat_id,
            project_id=project_id,
            tool_name=tool_name,
            provider=provider,
            display_name=display_name,
            supports_model=supports_model,
        )
        elements.append({"tag": "markdown", "content": "**模型族**"})
        elements.append(
            self._select_static(
                placeholder="选择模型",
                value=group_value,
                options=[self._select_option(g["label"], g["key"]) for g in groups],
                initial_option=selected_group_key,
            )
        )

        variants = list(selected_group["variants"])
        profiles = self._ordered_unique((v["profile"] for v in variants), kind="profile")
        profile_default = (
            default_profile
            if selected_group_key == default_group
            else group_default_profile
        )
        if self.pending_model_profile in profiles:
            selected_profile = self.pending_model_profile
        elif profile_default in profiles:
            selected_profile = profile_default
        else:
            selected_profile = profiles[0]
        if len(profiles) > 1:
            profile_action = (
                WORKFLOW_REVIEW_SELECT_MODEL_PROFILE
                if is_review
                else WORKFLOW_ORCHESTRATOR_SELECT_MODEL_PROFILE
            )
            profile_value = self._make_button_value(
                action=profile_action,
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
                tool_name=tool_name,
                provider=provider,
                display_name=display_name,
                supports_model=supports_model,
                model_group=selected_group_key,
            )
            elements.append({"tag": "markdown", "content": "**Profile**"})
            elements.append(
                self._select_static(
                    placeholder="选择 Profile",
                    value=profile_value,
                    options=[self._select_option(self._profile_label(p), p) for p in profiles],
                    initial_option=selected_profile,
                )
            )

        profile_variants = [v for v in variants if v["profile"] == selected_profile]
        efforts = self._ordered_unique((v["effort"] for v in profile_variants), kind="effort")
        effort_default = (
            default_effort
            if selected_group_key == default_group
            and selected_profile == default_profile
            else (
                group_default_effort
                if selected_profile == group_default_profile
                else None
            )
        )
        if self.pending_model_effort in efforts:
            selected_effort = self.pending_model_effort
        elif effort_default in efforts:
            selected_effort = effort_default
        else:
            selected_effort = efforts[0]
        if len(efforts) > 1:
            effort_action = (
                WORKFLOW_REVIEW_SELECT_MODEL_EFFORT
                if is_review
                else WORKFLOW_ORCHESTRATOR_SELECT_MODEL_EFFORT
            )
            effort_value = self._make_button_value(
                action=effort_action,
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
                tool_name=tool_name,
                provider=provider,
                display_name=display_name,
                supports_model=supports_model,
                model_group=selected_group_key,
                model_profile=selected_profile,
            )
            elements.append({"tag": "markdown", "content": "**Effort**"})
            elements.append(
                self._select_static(
                    placeholder="选择 Effort",
                    value=effort_value,
                    options=[self._select_option(self._effort_label(e), e) for e in efforts],
                    initial_option=selected_effort,
                )
            )

        selected_variant = self._choose_variant(
            variants,
            profile=selected_profile,
            effort=selected_effort,
        )
        if not selected_variant:
            selected_variant = variants[0]
        model_name = str(selected_variant["name"])
        add_value = self._make_button_value(
            action=model_action,
            session_key=session_key,
            is_review=is_review,
            chat_id=chat_id,
            project_id=project_id,
            tool_name=tool_name,
            provider=provider,
            display_name=display_name,
            supports_model=supports_model,
            model_name=model_name,
            name=model_name,
            model_group=selected_group_key,
            model_profile=selected_variant["profile"],
            model_effort=selected_variant["effort"],
        )
        add_label = "添加此组合" if is_review else "设为主编排 Agent"
        elements.extend(
            build_responsive_button_row([
                {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": self._button_label(f"{add_label}: {model_name}"),
                    },
                    "type": "primary",
                    "value": add_value,
                    "behaviors": [{"type": "callback", "value": add_value}],
                }
            ])
        )

    @staticmethod
    def _build_model_groups(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return model_cascade.build_model_groups(models)

    @staticmethod
    def _split_model_variant(model_name: str) -> tuple[str, tuple[str, ...]]:
        return model_cascade.split_model_variant(model_name)

    @staticmethod
    def _dimensions_from_tokens(tokens: tuple[str, ...]) -> tuple[str, str]:
        return model_cascade.dimensions_from_tokens(tokens)

    @staticmethod
    def _ordered_unique(values: Any, *, kind: str) -> list[str]:
        return model_cascade.ordered_unique(values, kind=kind)

    @staticmethod
    def _choose_variant(
        variants: list[dict[str, Any]],
        *,
        profile: str,
        effort: str,
    ) -> dict[str, Any] | None:
        return model_cascade.choose_variant(variants, profile=profile, effort=effort)


    @staticmethod
    def _select_static(
        *,
        placeholder: str,
        value: dict[str, Any],
        options: list[dict[str, Any]],
        initial_option: str | None = None,
    ) -> dict[str, Any]:
        select = {
            "tag": "select_static",
            "placeholder": {"tag": "plain_text", "content": placeholder},
            "value": value,
            "options": options,
        }
        if initial_option:
            select["initial_option"] = initial_option
        return select

    @staticmethod
    def _select_option(label: str, value: str) -> dict[str, Any]:
        return {
            "text": {"tag": "plain_text", "content": SelectionFlowController._select_label(label)},
            "value": value,
        }

    @staticmethod
    def _select_label(text: str) -> str:
        label = str(text or "").strip()
        if len(label) <= _SELECT_LABEL_MAX_CHARS:
            return label
        return label[: _SELECT_LABEL_MAX_CHARS - 1].rstrip() + "…"

    @staticmethod
    def _label_with_description(name: object, description: object = "") -> str:
        label = str(name or "").strip()
        desc = str(description or "").strip()
        if desc:
            return f"{label} ({desc})"
        return label

    @staticmethod
    def _profile_label(profile: str) -> str:
        if profile == "standard":
            return "standard"
        return profile

    @staticmethod
    def _effort_label(effort: str) -> str:
        if effort == "default":
            return "default"
        return effort

    def _make_button_value(
        self,
        *,
        action: str,
        session_key: str,
        is_review: bool,
        chat_id: str = "",
        project_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a button ``value`` dict matching the expected schema.

        The shape here mirrors ``_WORKFLOW_BUTTON_FIELDS`` in
        ``src/card/events/payloads.py`` so downstream filter / dispatch
        code continues to work unchanged.
        """
        value: dict[str, Any] = {"action": action}
        if session_key:
            value["engine_session_key"] = session_key
        if chat_id:
            value["chat_id"] = chat_id
        if project_id:
            value["project_id"] = project_id
        # Attach caller-provided fields — each maps to a known schema key
        # Boolean fields with semantic meaning when False (like supports_model)
        # must be preserved; only drop None and generic False values.
        _preserve_false_keys = {"supports_model"}
        for k, v in kwargs.items():
            if v is None:
                continue
            if v is False and k not in _preserve_false_keys:
                continue
            value[k] = v
        # Always preserve use_default_model when True
        if kwargs.get("use_default_model"):
            value["use_default_model"] = True
        return value

    @staticmethod
    def _selection_label(sel: dict[str, Any]) -> str:
        name = str(sel.get("display_name") or sel.get("tool_name", ""))
        if not name:
            name = "(unknown)"
        if sel.get("use_default_model") or not sel.get("model_name"):
            return f"`{name}` (默认模型)"
        return f"`{name}` · {sel['model_name']}"

    @staticmethod
    def _button_label(text: str) -> str:
        label = str(text or "").strip()
        if len(label) <= _MODEL_BUTTON_LABEL_MAX_CHARS:
            return label
        return label[: _MODEL_BUTTON_LABEL_MAX_CHARS - 3] + "..."

    def _model_page_buttons(
        self,
        *,
        action: str,
        session_key: str,
        is_review: bool,
        chat_id: str,
        project_id: str,
        tool_name: str,
        provider: str,
        display_name: str,
        supports_model: bool,
        page: int,
        total_pages: int,
    ) -> list[dict[str, Any]]:
        if total_pages <= 1:
            return []
        buttons: list[dict[str, Any]] = []
        if page > 0:
            prev_value = self._make_button_value(
                action=action,
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
                tool_name=tool_name,
                provider=provider,
                display_name=display_name,
                supports_model=supports_model,
                model_page=page - 1,
            )
            buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": "上一页"},
                "type": "default",
                "value": prev_value,
                "behaviors": [{"type": "callback", "value": prev_value}],
            })
        if page + 1 < total_pages:
            next_value = self._make_button_value(
                action=action,
                session_key=session_key,
                is_review=is_review,
                chat_id=chat_id,
                project_id=project_id,
                tool_name=tool_name,
                provider=provider,
                display_name=display_name,
                supports_model=supports_model,
                model_page=page + 1,
            )
            buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": "下一页"},
                "type": "default",
                "value": next_value,
                "behaviors": [{"type": "callback", "value": next_value}],
            })
        return buttons
