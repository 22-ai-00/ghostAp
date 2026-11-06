"""Tests for task factory helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.card.task_registry import tasks_from_plan_entries

# ---------------------------------------------------------------------------
# Fake models for testing (avoid importing full ACP/Spec dependencies)
# ---------------------------------------------------------------------------


@dataclass
class FakePlanEntry:
    content: str
    priority: str = "medium"
    status: str = "pending"


@dataclass
class FakePlanInfo:
    entries: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tests: tasks_from_plan_entries
# ---------------------------------------------------------------------------


class TestTasksFromPlanEntries:
    def test_basic_conversion(self):
        entries = [
            FakePlanEntry(content="Analyze requirements"),
            FakePlanEntry(content="Write code"),
            FakePlanEntry(content="Run tests"),
        ]
        result = tasks_from_plan_entries(entries)
        assert len(result) == 3
        assert result[0] == {"task_id": "step_0", "name": "Analyze requirements", "status": "pending"}
        assert result[1] == {"task_id": "step_1", "name": "Write code", "status": "pending"}
        assert result[2] == {"task_id": "step_2", "name": "Run tests", "status": "pending"}

    def test_preserves_status(self):
        entries = [
            FakePlanEntry(content="Done task", status="completed"),
            FakePlanEntry(content="Active task", status="in_progress"),
        ]
        result = tasks_from_plan_entries(entries)
        assert result[0]["status"] == "completed"
        assert result[1]["status"] == "in_progress"

    def test_skips_empty_content(self):
        entries = [
            FakePlanEntry(content=""),
            FakePlanEntry(content="  "),
            FakePlanEntry(content="Valid task"),
        ]
        result = tasks_from_plan_entries(entries)
        assert len(result) == 1
        assert result[0]["name"] == "Valid task"

    def test_truncates_long_names(self):
        entries = [FakePlanEntry(content="x" * 200)]
        result = tasks_from_plan_entries(entries)
        assert len(result[0]["name"]) == 120

    def test_structured_json_content_uses_readable_name(self):
        payload = json.dumps({
            "command": ["/usr/bin/zsh", "-lc", "nl -ba src/card/orchestrator.py"],
            "parsed_cmd": [{
                "type": "read",
                "path": "src/card/orchestrator.py",
            }],
            "stdout": "large output should not become a task name",
        })
        entries = [FakePlanEntry(content=payload)]

        result = tasks_from_plan_entries(entries)

        assert result[0]["name"] == "读取 src/card/orchestrator.py"

    def test_invalid_status_defaults_to_pending(self):
        entries = [FakePlanEntry(content="Task", status="unknown_status")]
        result = tasks_from_plan_entries(entries)
        assert result[0]["status"] == "pending"

    def test_empty_list(self):
        result = tasks_from_plan_entries([])
        assert result == []
