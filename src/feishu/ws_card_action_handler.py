"""Helpers for Feishu card action callback inspection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


def bind_managed_trust_revisions(
    card: dict[str, Any],
    *,
    group_revision: int,
    grant_revision: int,
) -> dict[str, Any]:
    """Copy a card and stamp every callback value with issued revisions."""

    if type(group_revision) is not int or group_revision < 1:
        raise ValueError("group_revision must be positive")
    if type(grant_revision) is not int or grant_revision < 1:
        raise ValueError("grant_revision must be positive")
    bound = deepcopy(card)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in tuple(value.items()):
                if (
                    key == "value"
                    and isinstance(child, dict)
                    and ("action" in child or "action_id" in child)
                ):
                    child["group_revision"] = group_revision
                    child["grant_revision"] = grant_revision
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(bound)
    return bound


def _extract_behavior_value(behavior: Any) -> Any:
    """Extract value from a behavior object, supporting both object and Mapping forms.

    Handles both:
    - Object form: getattr(behavior, "value", None)
    - Mapping form: behavior.get("value") if isinstance(behavior, Mapping)
    """
    if isinstance(behavior, Mapping):
        return behavior.get("value")
    return getattr(behavior, "value", None)

# NOTE: This set must remain in sync with the action registrations in
# ``src/feishu/action_registry.py`` and the ``WORKFLOW_*`` constants defined in
# ``src/card/actions/dispatch.py``.  Every newly registered workflow action must
# also be added here so ``CardActionInspector.is_system_action`` recognises it
# and routes it through the normal card-action pipeline.
SYSTEM_CARD_ACTIONS = {
    "show_status",
    "switch_project",
    "show_board",
    "refresh_board",
    "show_detail",
    "new_project_prompt",
    "select_acp_tool",
    "select_acp_model",
    "refresh_acp_models",
    "slock_new_role_select_tool",
    "slock_new_role_select_tool_dropdown",
    "slock_new_role_select_model",
    "load_more",
    "load_prev",
    "show_acp_menu",
    "show_help_menu",
    "force_release_repo_lock",
    "confirm_lock",
    "cancel_lock",
    "confirm_force_release",
    "cancel_force_release",
    "enter_deep_prompt",
    "continue_degraded",
    "show_error_details",
    "retry_original",
    "help_category",
    "engine_stop",
    "deep_pause",
    "deep_stop",
    "deep_resume",
    "spec_pause",
    "spec_stop",
    "spec_resume",
    "spec_skip_retry",
    "spec_review_use_auto",
    "spec_review_finish_selection",
    "spec_review_select_tool",
    "spec_review_select_model",
    "spec_review_remove_item",
    "spec_review_clear_items",
    "spec_restore_run",
    "show_spec_review_menu",
    "workflow_confirm_start",
    "workflow_cancel",
    "workflow_stop_running",
    "workflow_select_tool",
    "workflow_confirm_tools",
    "workflow_regenerate_script",
    "workflow_fill_missing_tools",
    "workflow_back_to_tools",
    "workflow_view_workflow_ref",
    "workflow_remove_workflow_ref",
    "workflow_add_workflow_ref",
    "show_workflow_menu",
    "workflow_list_templates",
    "workflow_show_help",
    # Workflow orchestrator selection (Step 1 of 2-step flow, combined card)
    "workflow_orchestrator_select_tool",
    "workflow_orchestrator_select_model_group",
    "workflow_orchestrator_select_model_profile",
    "workflow_orchestrator_select_model_effort",
    "workflow_orchestrator_select_model",
    "workflow_orchestrator_remove",
    "workflow_orchestrator_clear",
    "workflow_orchestrator_finish",
    # Workflow review agent selection (Step 2 of 2-step flow)
    "workflow_review_select_tool",
    "workflow_review_select_model_group",
    "workflow_review_select_model_profile",
    "workflow_review_select_model_effort",
    "workflow_review_select_model",
    "workflow_review_finish",
    "workflow_review_remove",
    "workflow_review_clear",
    "workflow_review_toggle_auto",
}


class CardActionFailureAction(str, Enum):
    ACK_AND_IGNORE = "ack_and_ignore"
    REPLY_FAILURE_CARD = "reply_failure_card"
    RAISE = "raise"


@dataclass(frozen=True)
class CardActionErrorClassification:
    action: CardActionFailureAction
    phase: str
    user_reachable: bool


def classify_card_action_error(error: Exception, *, phase: str) -> CardActionErrorClassification:
    if phase in {"payload_parse", "dedup"}:
        return CardActionErrorClassification(CardActionFailureAction.ACK_AND_IGNORE, phase, False)
    if phase == "dispatch":
        return CardActionErrorClassification(CardActionFailureAction.REPLY_FAILURE_CARD, phase, True)
    return CardActionErrorClassification(CardActionFailureAction.RAISE, phase, False)


class CardActionInspector:
    """Pure helpers for extracting stable card action callback fields."""

    @classmethod
    def value_dict(cls, action: Any) -> dict[str, Any]:
        # Schema 2.0: prefer behaviors[0].value, fallback to legacy action.value
        value_raw = getattr(action, "value", None)
        behaviors = getattr(action, "behaviors", None)
        if isinstance(behaviors, list) and behaviors:
            first_behavior = behaviors[0]
            behavior_value = _extract_behavior_value(first_behavior)
            if behavior_value is not None:
                value_raw = behavior_value
        if isinstance(value_raw, dict):
            return value_raw
        if isinstance(value_raw, str):
            try:
                parsed = json.loads(value_raw)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def action_type(cls, action: Any) -> str:
        return str(cls.value_dict(action).get("action", "") or "")

    @classmethod
    def project_id(cls, action: Any) -> Optional[str]:
        project_id = cls.value_dict(action).get("project_id")
        return project_id if isinstance(project_id, str) and project_id else None

    @classmethod
    def trust_revisions(cls, action: Any) -> tuple[int | None, int | None]:
        """Extract an optional immutable managed-group revision snapshot."""

        value = cls.value_dict(action)

        def revision(name: str) -> int | None:
            item = value.get(name)
            return item if type(item) is int and item > 0 else None

        return revision("group_revision"), revision("grant_revision")

    @classmethod
    def is_system_action(cls, action: Any) -> bool:
        return cls.action_type(action) in SYSTEM_CARD_ACTIONS

    @classmethod
    def dedup_fingerprint(cls, action: Any) -> str:
        # Schema 2.0: prefer behaviors[0].value, fallback to legacy action.value
        value_raw = getattr(action, "value", None)
        behaviors = getattr(action, "behaviors", None)
        if isinstance(behaviors, list) and behaviors:
            first_behavior = behaviors[0]
            behavior_value = _extract_behavior_value(first_behavior)
            if behavior_value is not None:
                value_raw = behavior_value
        payload: dict[str, Any] = {"value": cls._normalize_value(value_raw)}
        for attr in ("option", "options", "form_value", "input_value"):
            extra = getattr(action, attr, None)
            if isinstance(extra, (str, int, float, bool, list, tuple, dict)):
                payload[attr] = cls._normalize_value(extra)

        try:
            canonical = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            canonical = str(payload)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
