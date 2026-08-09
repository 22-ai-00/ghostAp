"""API contract tests for the workflow subsystem.

Covers three guarantees the rest of the codebase relies on:

1. Every ``WORKFLOW_`` card action defined in
   ``src/card/actions/dispatch.py`` is also listed in
   ``SYSTEM_CARD_ACTIONS`` in ``src/feishu/ws_card_action_handler.py`` so
   the Feishu card-action pipeline accepts it.

2. Every ``WORKFLOW_`` card action registered in ``init_action_registry``
   dispatches to a handler method that accepts the standard
   ``(message_id, chat_id, project_id, value)`` positional signature. This
   is the **dispatch contract** — if someone adds a new workflow action but
   forgets to wire up its handler signature, the test fails loudly *before*
   the broken code hits production.

3. ``build_script_gen_prompt`` is resilient to prompt-template mutations.
   If the sentinel marker that marks the budget/agent-capability insertion
   point is missing — because someone edited the template, or a test
   monkey-patched it — the function must still return a valid string and
   must **not** raise ``ValueError``.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

from src.card.actions import dispatch as action_ids
from src.feishu import action_registry as _action_registry_module
from src.feishu.handlers.workflow import WorkflowHandler
from src.feishu.ws_card_action_handler import SYSTEM_CARD_ACTIONS
from src.workflow_engine import script_gen

# Actions whose primary purpose is navigation / menu display rather than a
# card-bound system action.  These are intentionally allowed to be absent from
# ``SYSTEM_CARD_ACTIONS`` because they are handled by prefix / menu dispatch
# instead of the strict card-action gate.
_NAV_EXCEPTIONS = frozenset(
    {
        action_ids.SHOW_WORKFLOW_MENU,
        action_ids.WORKFLOW_LIST_TEMPLATES,
        action_ids.WORKFLOW_SHOW_HELP,
    }
)


def _collect_workflow_constants() -> list[str]:
    """Return all ``WORKFLOW_*`` / ``SHOW_WORKFLOW_*`` string constants."""
    values: list[str] = []
    for name, value in inspect.getmembers(action_ids):
        if not (name.startswith("WORKFLOW_") or name.startswith("SHOW_WORKFLOW_")):
            continue
        if isinstance(value, str) and value:
            values.append(value)
    # Deduplicate while preserving deterministic order.
    seen: set[str] = set()
    ordered: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def test_all_workflow_constants_are_registered_in_system_actions() -> None:
    """Every WORKFLOW_* action id must be in SYSTEM_CARD_ACTIONS."""
    workflow_constants = _collect_workflow_constants()
    assert workflow_constants, "Expected at least one WORKFLOW_* constant to exist"

    missing = [
        action_id
        for action_id in workflow_constants
        if action_id not in SYSTEM_CARD_ACTIONS
        and action_id not in _NAV_EXCEPTIONS
    ]
    assert not missing, (
        "These WORKFLOW_* action ids are defined in src/card/actions/dispatch.py "
        "but missing from SYSTEM_CARD_ACTIONS in src/feishu/ws_card_action_handler.py: "
        f"{missing!r}"
    )


# ---------------------------------------------------------------------------
# Contract 2: every WORKFLOW_ action maps to a 4-positional-arg handler
# ---------------------------------------------------------------------------


def _collect_workflow_action_ids_only() -> list[str]:
    """Return every ``WORKFLOW_*`` action id (excluding SHOW_WORKFLOW_*)."""
    ids: list[str] = []
    for name in dir(action_ids):
        if name.startswith("WORKFLOW_"):
            value = getattr(action_ids, name)
            if isinstance(value, str):
                ids.append(value)
    assert ids, "No WORKFLOW_* action IDs found — did the module structure change?"
    return sorted(ids)




def _collect_registered_workflow_handlers() -> dict[str, Any]:
    """Capture exact workflow registrations without constructing a WS client."""
    workflow_action_ids = frozenset(_collect_workflow_action_ids_only())
    handlers: dict[str, Any] = {}
    client = MagicMock()

    def register(
        handler: Any,
        *,
        exact: str | None = None,
        prefix: str | None = None,
    ) -> None:
        del prefix
        if exact in workflow_action_ids:
            handlers[exact] = handler

    client._register_action.side_effect = register
    _action_registry_module.register_programming_mode_actions(client)
    return handlers


def _resolve_handler_method(registration: Any):
    """Inspect the registration callable to find the WorkflowHandler method.

    Accepted patterns:

    * ``lambda mid, cid, pid, val: client._handle_workflow_X(mid, cid, pid, val)``
      — extract ``_handle_workflow_X`` and return the method on
      ``WorkflowHandler`` (which strips the leading underscore).

    Returns the underlying method if traceable, otherwise ``None``.
    """
    import re

    try:
        source = inspect.getsource(registration)
    except (OSError, TypeError):
        return None

    match = re.search(r"(?P<name>_handle_workflow_\w+)\s*\(", source)
    if not match:
        return None
    method_name = match.group("name")
    public_name = method_name.lstrip("_")
    return getattr(WorkflowHandler, public_name, None)


class TestWorkflowActionSignatureContract:
    """All workflow actions must uniformly accept (message_id, chat_id, project_id, value)."""

    def test_workflow_actions_have_handlers_with_four_positional_args(self) -> None:
        handlers = _collect_registered_workflow_handlers()
        for action_id in _collect_workflow_action_ids_only():
            assert action_id in handlers, (
                f"Action {action_id!r} is exported but not registered. Add a "
                "``client._register_action(lambda ..., exact=...)`` call in "
                "``register_programming_mode_actions``."
            )

            registration = handlers[action_id]
            method = _resolve_handler_method(registration)
            assert method is not None, (
                f"Could not resolve the underlying WorkflowHandler method for "
                f"{action_id!r}. Is the registration lambda shaped like "
                "``lambda mid, cid, pid, val: client._handle_workflow_X(mid, cid, pid, val)``?"
            )

            sig = inspect.signature(method)
            params = [
                name
                for name, parameter in sig.parameters.items()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            positional = [parameter for parameter in params if parameter != "self"]
            assert len(positional) == 4, (
                f"{method.__name__} has {len(positional)} positional args "
                f"({positional!r}); expected 4: "
                "(message_id, chat_id, project_id, value). "
                "Add a ``project_id`` slot even if the handler does not use it — "
                "this preserves the uniform dispatch contract."
            )
            assert positional == ["message_id", "chat_id", "project_id", "value"], (
                f"{method.__name__} positional args are {positional!r}; "
                "expected ['message_id', 'chat_id', 'project_id', 'value']."
            )


# ---------------------------------------------------------------------------
# Contract 3: build_script_gen_prompt is resilient to sentinel mutations
# ---------------------------------------------------------------------------


class TestScriptGenPromptMarkerFallback:
    """The prompt builder must never raise ValueError over a missing marker."""

    def test_prompt_contains_budget_and_agent_sections_when_sentinel_present(
        self,
    ) -> None:
        """Baseline: with the sentinel intact, agent capability section is injected."""
        prompt = script_gen.build_script_gen_prompt(
            requirement="Build a parser",
            available_tools=["coco", "claude"],
            orchestrator_agent="coco",
        )

        assert isinstance(prompt, str)
        assert prompt.strip()
        assert "## User Requirement" in prompt
        user_req_idx = prompt.find("## User Requirement")
        assert "主编排 Agent 能力" in prompt[:user_req_idx]
        assert "SENTINEL" not in prompt
        assert "USER_REQUIREMENT_INSERT_POINT" not in prompt

    def test_prompt_does_not_raise_when_sentinel_missing(self) -> None:
        """Mutate the template so the sentinel is gone — still returns a valid string.

        Previously this raised ``ValueError`` because the code used
        ``str.index('## User Requirement')`` without a guard. Now it should
        fall back to appending the extra sections at the end.
        """
        original_template = script_gen._SCRIPT_GEN_PROMPT_TEMPLATE
        try:
            script_gen._SCRIPT_GEN_PROMPT_TEMPLATE = (
                "# Custom Header\n\n"
                "## User Requirement\n\n{requirement}\n\n"
                "## Available Resources\n\n{tools_list}\n\n"
                "## Output Format\n\nGenerate a script.\n"
            )

            prompt = script_gen.build_script_gen_prompt(
                requirement="Do X",
                available_tools=["coco"],
                orchestrator_agent="claude",
            )

            assert isinstance(prompt, str)
            assert prompt.strip()
            assert "主编排 Agent 能力" in prompt
            assert "ValueError" not in prompt
        finally:
            script_gen._SCRIPT_GEN_PROMPT_TEMPLATE = original_template

    def test_prompt_without_budget_tokens_still_injects_agent_section(self) -> None:
        """With new API: budget is removed; only orchestrator agent section is injected.

        Verifies the injection path works without budget concepts —
        the prompt builder now only has requirement + available_tools +
        orchestrator_agent parameters.
        """
        prompt = script_gen.build_script_gen_prompt(
            requirement="Analyze logs",
            available_tools=["coco"],
            orchestrator_agent="claude",
        )

        assert isinstance(prompt, str)
        assert prompt.strip()
        assert "主编排 Agent 能力" in prompt
        assert "预算" not in prompt
