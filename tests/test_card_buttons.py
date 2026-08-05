"""Tests for card buttons module (AC21)."""

from __future__ import annotations


def test_workflow_action_ids_in_valid_keys():
    """AC21: Workflow action_ids 包含在 _valid_keys 集合中。"""
    from src.card.actions.dispatch import (
        SHOW_WORKFLOW_MENU,
        WORKFLOW_CANCEL,
        WORKFLOW_CONFIRM_START,
        WORKFLOW_CONFIRM_TOOLS,
        WORKFLOW_LIST_TEMPLATES,
        WORKFLOW_ORCHESTRATOR_FINISH,
        WORKFLOW_ORCHESTRATOR_SELECT_MODEL,
        WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
        WORKFLOW_REGENERATE_SCRIPT,
        WORKFLOW_REVIEW_FINISH,
        WORKFLOW_REVIEW_SELECT_MODEL,
        WORKFLOW_REVIEW_SELECT_TOOL,
        WORKFLOW_SELECT_TOOL,
        WORKFLOW_SHOW_HELP,
    )
    from src.card.render.buttons import _valid_keys

    workflow_action_ids = [
        WORKFLOW_CANCEL,
        WORKFLOW_CONFIRM_TOOLS,
        WORKFLOW_CONFIRM_START,
        WORKFLOW_SELECT_TOOL,
        WORKFLOW_REGENERATE_SCRIPT,
        WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
        WORKFLOW_ORCHESTRATOR_SELECT_MODEL,
        WORKFLOW_ORCHESTRATOR_FINISH,
        WORKFLOW_REVIEW_SELECT_TOOL,
        WORKFLOW_REVIEW_SELECT_MODEL,
        WORKFLOW_REVIEW_FINISH,
        SHOW_WORKFLOW_MENU,
        WORKFLOW_LIST_TEMPLATES,
        WORKFLOW_SHOW_HELP,
    ]

    for action_id in workflow_action_ids:
        assert action_id in _valid_keys, f"Workflow action_id {action_id} not in _valid_keys"


def test_import_with_error_on_warning():
    """AC21: 使用 -W error::RuntimeWarning 运行时导入不报错。"""
    # This test is more of a documentation — the actual runtime check
    # is done by running pytest with -W error::RuntimeWarning
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "-W", "error::RuntimeWarning",
            "-c", "from src.card.render import buttons; print('OK')"
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Import failed with RuntimeWarning treated as error. "
        f"Stderr: {result.stderr}"
    )
    assert "OK" in result.stdout
