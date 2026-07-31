from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.worktree_engine.models import WorktreeUnit, WorktreeUnitStatus
from src.worktree_engine.review_adapter import (
    WorktreeReviewAdapter,
    WorktreeReviewVerdict,
)


class _PromptResult:
    def __init__(
        self,
        text: str,
        *,
        stop_reason: str = "end_turn",
        tool_results: list[dict] | None = None,
    ):
        self.text = text
        self.stop_reason = stop_reason
        self.tool_results = (
            tool_results
            if tool_results is not None
            else [
                {
                    "kind": "execute",
                    "data": {"command": "uv run pytest -q", "exit_code": 0},
                }
            ]
        )


class _ReviewSession:
    def __init__(self, response: str, prompts: list[str]):
        self._response = response
        self._prompts = prompts

    def send_prompt(self, prompt: str, *, on_event=None, timeout=None):
        del on_event, timeout
        self._prompts.append(prompt)
        return _PromptResult(self._response)

    def close(self):
        return None


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _changed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "review-wt"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "review@example.invalid")
    _git(repo, "config", "user.name", "Review Test")
    target = repo / "feature.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "base")
    target.write_text("VALUE = 2\n", encoding="utf-8")
    return repo


def test_review_consumes_real_findings_before_completion(tmp_path):
    repo = _changed_repo(tmp_path)
    prompts: list[str] = []
    response = """{
      "verdict": "PASS",
      "summary": "diff and verification are sound",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": [{"severity": "observation", "message": "value change verified", "evidence": "+VALUE = 2"}]
    }"""
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession(response, prompts)
    )
    unit = WorktreeUnit(
        unit_id="u1",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        task_title="change the value",
        summary="implementation complete; targeted test passed",
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
        metadata={"test_results": ["uv run pytest tests/test_feature.py -q: 12 passed"]},
    )

    outcome = adapter.review_unit(goal="change VALUE to 2", unit=unit, timeout=5)

    assert outcome.verdict is WorktreeReviewVerdict.PASS
    assert outcome.passed is True
    assert outcome.findings
    assert outcome.tests
    assert outcome.evidence["diff_sha256"]
    assert outcome.evidence["input_test_count"] == 1
    prompt = prompts[0]
    assert "-VALUE = 1" in prompt
    assert "+VALUE = 2" in prompt
    assert "12 passed" in prompt
    assert "implementation complete" in prompt


def test_empty_findings_are_not_treated_as_pass(tmp_path):
    repo = _changed_repo(tmp_path)
    response = """{
      "verdict": "PASS",
      "summary": "looks fine",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": []
    }"""
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession(response, [])
    )
    unit = WorktreeUnit(
        unit_id="u1",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="change VALUE to 2", unit=unit, timeout=5)

    assert outcome.verdict is WorktreeReviewVerdict.INCONCLUSIVE
    assert outcome.passed is False
    assert outcome.error_code == "empty_findings"


def test_committed_branch_diff_is_included_in_review_evidence(tmp_path):
    repo = _changed_repo(tmp_path)
    _git(repo, "checkout", "--", "feature.py")
    base_branch = _git_output(repo, "branch", "--show-current")
    _git(repo, "checkout", "-b", "review-feature")
    (repo / "feature.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "change value")
    prompts: list[str] = []
    response = """{
      "verdict": "PASS",
      "summary": "committed change verified",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": [{"severity": "observation", "message": "commit verified", "evidence": "+VALUE = 3"}]
    }"""
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession(response, prompts)
    )
    unit = WorktreeUnit(
        unit_id="committed",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
    )

    outcome = adapter.review_unit(
        goal="change VALUE to 3",
        unit=unit,
        timeout=5,
        base_branch=base_branch,
    )

    assert outcome.passed is True
    assert "+VALUE = 3" in prompts[0]


def test_non_json_review_is_inconclusive(tmp_path):
    repo = _changed_repo(tmp_path)
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession("not json", [])
    )
    unit = WorktreeUnit(
        unit_id="invalid-review",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="change VALUE", unit=unit, timeout=5)

    assert outcome.verdict is WorktreeReviewVerdict.INCONCLUSIVE
    assert outcome.passed is False
    assert outcome.error_code == "review_error"


def test_unknown_review_stop_reason_is_inconclusive(tmp_path):
    repo = _changed_repo(tmp_path)

    class _IncompleteReviewSession(_ReviewSession):
        def send_prompt(self, prompt: str, *, on_event=None, timeout=None):
            del prompt, on_event, timeout
            return _PromptResult(self._response, stop_reason="max_tokens")

    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _IncompleteReviewSession("{}", [])
    )
    unit = WorktreeUnit(
        unit_id="incomplete-review",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="change VALUE", unit=unit, timeout=5)

    assert outcome.verdict is WorktreeReviewVerdict.INCONCLUSIVE
    assert outcome.passed is False
    assert outcome.error_code == "review_max_tokens"


def test_pass_requires_test_command_and_output_evidence(tmp_path):
    repo = _changed_repo(tmp_path)
    response = """{
      "verdict": "PASS",
      "summary": "claimed pass without output",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": ""}],
      "findings": [{"severity": "observation", "message": "diff checked", "evidence": "+VALUE = 2"}]
    }"""
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession(response, [])
    )
    unit = WorktreeUnit(
        unit_id="missing-test-output",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="change VALUE", unit=unit, timeout=5)

    assert outcome.verdict is WorktreeReviewVerdict.INCONCLUSIVE
    assert outcome.passed is False
    assert outcome.error_code == "invalid_test_evidence"


def test_unsupported_blocker_without_evidence_cannot_be_downgraded(tmp_path):
    repo = _changed_repo(tmp_path)
    response = """{
      "verdict": "PASS",
      "summary": "contradictory review",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": [
        {"severity": "blocker", "message": "possible data loss", "evidence": ""},
        {"severity": "observation", "message": "diff checked", "evidence": "+VALUE = 2"}
      ]
    }"""
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession(response, [])
    )
    unit = WorktreeUnit(
        unit_id="blocker",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="change VALUE", unit=unit, timeout=5)

    assert outcome.verdict is WorktreeReviewVerdict.FAIL
    assert outcome.passed is False
    assert outcome.blockers[0]["severity"] == "blocker"


def test_review_diff_includes_files_inside_untracked_directories(tmp_path):
    repo = _changed_repo(tmp_path)
    (repo / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    package = repo / "new_package"
    package.mkdir()
    (package / "feature.py").write_text("NEW_VALUE = 3\n", encoding="utf-8")
    prompts: list[str] = []
    response = """{
      "verdict": "PASS",
      "summary": "new file checked",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": [{"severity": "observation", "message": "new file checked", "evidence": "+NEW_VALUE = 3"}]
    }"""
    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession(response, prompts)
    )
    unit = WorktreeUnit(
        unit_id="untracked-directory",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="add NEW_VALUE", unit=unit, timeout=5)

    assert outcome.passed is True
    assert "new_package/feature.py" in prompts[0]
    assert "+NEW_VALUE = 3" in prompts[0]


@pytest.mark.parametrize(
    ("executed_command", "exit_code", "expected_verdict", "expected_error"),
    [
        ("false", 1, WorktreeReviewVerdict.INCONCLUSIVE, "unverified_test_evidence"),
        ("uv run pytest -q", 1, WorktreeReviewVerdict.FAIL, "tests_failed"),
        (
            "uv run pytest -q",
            None,
            WorktreeReviewVerdict.INCONCLUSIVE,
            "unknown_test_exit_code",
        ),
    ],
)
def test_forged_test_pass_cannot_bypass_execution_provenance(
    tmp_path,
    executed_command,
    exit_code,
    expected_verdict,
    expected_error,
):
    repo = _changed_repo(tmp_path)
    response = """{
      "verdict": "PASS",
      "summary": "claimed test pass",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": [{"severity": "observation", "message": "diff checked", "evidence": "+VALUE = 2"}]
    }"""

    class _ForgedReviewSession(_ReviewSession):
        def send_prompt(self, prompt: str, *, on_event=None, timeout=None):
            del prompt, on_event, timeout
            return _PromptResult(
                self._response,
                tool_results=[
                    {
                        "kind": "execute",
                        "data": {
                            "command": executed_command,
                            "exit_code": exit_code,
                        },
                    }
                ],
            )

    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ForgedReviewSession(response, [])
    )
    unit = WorktreeUnit(
        unit_id="forged-pass",
        provider="acp",
        tool_name="coco",
        worktree_path=str(repo),
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
    )

    outcome = adapter.review_unit(goal="change VALUE", unit=unit, timeout=5)

    assert outcome.verdict is expected_verdict
    assert outcome.passed is False
    assert outcome.error_code == expected_error


@pytest.mark.parametrize(
    ("exit_codes", "expected_verdict", "expected_passed"),
    [
        ([0, 1], WorktreeReviewVerdict.FAIL, False),
        ([1, 0], WorktreeReviewVerdict.PASS, True),
    ],
)
def test_latest_matching_test_execution_is_authoritative(
    tmp_path,
    exit_codes,
    expected_verdict,
    expected_passed,
):
    repo = _changed_repo(tmp_path)
    response = """{
      "verdict": "PASS",
      "summary": "ordered test executions",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "final run"}],
      "findings": [{"severity": "observation", "message": "diff checked", "evidence": "+VALUE = 2"}]
    }"""

    class _OrderedReviewSession(_ReviewSession):
        def send_prompt(self, prompt: str, *, on_event=None, timeout=None):
            del prompt, on_event, timeout
            return _PromptResult(
                self._response,
                tool_results=[
                    {
                        "kind": "execute",
                        "data": {
                            "command": "uv run pytest -q",
                            "exit_code": exit_code,
                        },
                    }
                    for exit_code in exit_codes
                ],
            )

    outcome = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _OrderedReviewSession(response, [])
    ).review_unit(
        goal="change VALUE",
        unit=WorktreeUnit(
            unit_id="ordered-tests",
            provider="acp",
            tool_name="coco",
            worktree_path=str(repo),
            status=WorktreeUnitStatus.COMPLETED,
            has_changes=True,
        ),
        timeout=5,
    )

    assert outcome.verdict is expected_verdict
    assert outcome.passed is expected_passed
    assert outcome.tests[0]["exit_code"] == exit_codes[-1]
