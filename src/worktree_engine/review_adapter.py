from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from ..acp.outcome import PromptOutcome, classify_prompt_result
from ..utils.errors import get_error_detail
from .models import WorktreeUnit

if TYPE_CHECKING:
    from ..agent_session import SyncSession

logger = logging.getLogger(__name__)

MAX_REVIEW_DIFF_CHARS = 100_000
MAX_REVIEW_TEXT_CHARS = 12_000


@dataclass(frozen=True)
class WorktreeReviewRole:
    role_id: str
    display_name: str
    blocking: bool = True


@dataclass
class WorktreeReviewPlan:
    roles: list[WorktreeReviewRole] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": [
                {
                    "role_id": role.role_id,
                    "display_name": role.display_name,
                    "blocking": role.blocking,
                }
                for role in self.roles
            ]
        }


class WorktreeReviewVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class WorktreeReviewOutcome:
    verdict: WorktreeReviewVerdict = WorktreeReviewVerdict.INCONCLUSIVE
    summary: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    unit_outcomes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.verdict is WorktreeReviewVerdict.PASS
            and bool(self.findings)
            and any(str(item.get("evidence") or "").strip() for item in self.findings)
            and bool(self.tests)
            and all(
                test.get("passed") is True
                and test.get("verified") is True
                and bool(str(test.get("command") or "").strip())
                and bool(str(test.get("evidence") or "").strip())
                for test in self.tests
            )
            and not self.blockers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "passed": self.passed,
            "summary": self.summary,
            "findings": list(self.findings),
            "tests": list(self.tests),
            "blockers": list(self.blockers),
            "observations": list(self.observations),
            "evidence": dict(self.evidence),
            "error_code": self.error_code,
            "unit_outcomes": list(self.unit_outcomes),
        }


ReviewSessionFactory = Callable[..., "SyncSession"]


def _truncate(value: object, limit: int = MAX_REVIEW_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated {len(text) - limit} chars]"


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("worktree review output must be a JSON object")
    return parsed


def _metadata_lines(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    if isinstance(value, Iterable):
        lines: list[str] = []
        for item in value:
            lines.extend(_metadata_lines(item))
        return lines
    text = str(value).strip()
    return [text] if text else []


def _capture_git_evidence(
    cwd: str,
    *,
    base_branch: str | None = None,
) -> tuple[str, list[str]]:
    root = Path(cwd)
    if not cwd or not root.is_dir():
        return "", []
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                cwd,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if status.returncode != 0:
            return "", []
        status_lines = [line for line in (status.stdout or "").splitlines() if line]
        touched_files = [line[3:].split(" -> ")[-1] for line in status_lines]

        working = subprocess.run(
            ["git", "-C", cwd, "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        diff_parts = [working.stdout or ""]
        if base_branch:
            committed = subprocess.run(
                [
                    "git",
                    "-C",
                    cwd,
                    "diff",
                    "--no-ext-diff",
                    "--binary",
                    f"{base_branch}...HEAD",
                    "--",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if committed.returncode == 0:
                diff_parts.insert(0, committed.stdout or "")
                committed_names = subprocess.run(
                    [
                        "git",
                        "-C",
                        cwd,
                        "diff",
                        "--name-only",
                        f"{base_branch}...HEAD",
                        "--",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                touched_files.extend(
                    line.strip()
                    for line in (committed_names.stdout or "").splitlines()
                    if line.strip()
                )
        for line in status_lines:
            if not line.startswith("?? "):
                continue
            relative = line[3:]
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            untracked = subprocess.run(
                [
                    "git",
                    "-C",
                    cwd,
                    "diff",
                    "--no-index",
                    "--binary",
                    "--",
                    "/dev/null",
                    relative,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            diff_parts.append(untracked.stdout or "")
            if sum(len(part) for part in diff_parts) >= MAX_REVIEW_DIFF_CHARS:
                break
        diff = "\n".join(part for part in diff_parts if part).strip()
        if len(diff) > MAX_REVIEW_DIFF_CHARS:
            diff = (
                f"{diff[:MAX_REVIEW_DIFF_CHARS]}\n"
                f"...[truncated {len(diff) - MAX_REVIEW_DIFF_CHARS} chars]"
            )
        return diff, list(dict.fromkeys(touched_files))[:100]
    except Exception as exc:
        logger.debug("[WorktreeReview] git evidence capture failed: %s", repr(exc))
        return "", []


def _normalize_findings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or item.get("recommendation") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        severity = str(item.get("severity") or "observation").strip().lower()
        if not message:
            continue
        if severity not in {"blocker", "major", "minor", "observation"}:
            severity = "blocker"
            evidence = evidence or "reviewer returned an unsupported severity"
            message = f"Invalid review severity: {message}"
        findings.append(
            {
                "severity": severity,
                "message": message,
                "evidence": evidence,
                "target": str(item.get("target") or "").strip(),
            }
        )
    return findings


def _normalize_tests(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tests: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or "").strip()
        evidence = str(item.get("evidence") or item.get("output") or "").strip()
        passed = item.get("passed")
        if not command and not evidence:
            continue
        tests.append(
            {
                "command": command,
                "passed": passed is True,
                "evidence": evidence,
            }
        )
    return tests


def _review_execution_evidence(result: object) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    for item in getattr(result, "tool_results", None) or ():
        if not isinstance(item, dict) or item.get("kind") != "execute":
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        command = str(data.get("command") or "").strip()
        if not command:
            continue
        executions.append(
            {
                "source": "review_session",
                "command": command,
                "exit_code": data.get("exit_code"),
            }
        )
    return executions


def _dispatcher_execution_evidence(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    executions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("source") != "worktree_dispatcher":
            continue
        command = str(item.get("command") or "").strip()
        if not command:
            continue
        executions.append(
            {
                "source": "worktree_dispatcher",
                "command": command,
                "exit_code": item.get("exit_code"),
            }
        )
    return executions


def _command_key(command: object) -> tuple[str, ...]:
    text = str(command or "").strip()
    if not text:
        return ()
    try:
        return tuple(shlex.split(text))
    except ValueError:
        return (" ".join(text.split()),)


def _bind_test_provenance(
    tests: list[dict[str, Any]],
    executions: list[dict[str, Any]],
) -> tuple[bool, bool, bool]:
    """Bind every claimed test to a matching structured command result.

    Returns ``(has_unverified, has_unknown_exit, has_failed_execution)``.
    """
    has_unverified = False
    has_unknown_exit = False
    has_failed_execution = False
    by_command: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for execution in executions:
        key = _command_key(execution.get("command"))
        if key:
            by_command.setdefault(key, []).append(execution)

    for test in tests:
        matches = by_command.get(_command_key(test.get("command")), [])
        if not matches:
            test["verified"] = False
            test["verification_error"] = "no_matching_execution"
            has_unverified = True
            continue
        latest = matches[-1]
        raw_code = latest.get("exit_code")
        if isinstance(raw_code, bool):
            exit_code = None
        else:
            try:
                exit_code = int(raw_code)
            except (TypeError, ValueError):
                exit_code = None
        if exit_code == 0:
            test["verified"] = True
            test["verification_source"] = latest.get("source", "")
            test["exit_code"] = 0
            continue
        test["verified"] = False
        if exit_code is not None:
            test["verification_error"] = "nonzero_exit_code"
            test["exit_code"] = exit_code
            has_failed_execution = True
        else:
            test["verification_error"] = "unknown_exit_code"
            has_unknown_exit = True
    return has_unverified, has_unknown_exit, has_failed_execution


class WorktreeReviewAdapter:
    """Independent, evidence-bound terminal review for Worktree units."""

    def __init__(self, *, session_factory: ReviewSessionFactory | None = None):
        self._session_factory = session_factory or self._default_session_factory

    @staticmethod
    def _default_session_factory(
        *,
        agent_type: str,
        cwd: str,
        model_name: str | None,
    ) -> "SyncSession":
        from ..agent_session import create_review_session

        return create_review_session(
            agent_type=agent_type,
            cwd=cwd,
            model_name=model_name,
        )

    @staticmethod
    def _agent_type(unit: WorktreeUnit) -> str:
        provider = str(unit.provider or "").strip().lower()
        tool_name = str(unit.tool_name or "").strip().lower()
        if provider == "ttadk":
            return f"ttadk_{tool_name}" if tool_name else "ttadk_coco"
        if provider == "cli":
            return tool_name or "claude"
        return tool_name or "coco"

    def plan_roles(self, *, goal: str, changed_files: list[str]) -> WorktreeReviewPlan:
        roles = [
            WorktreeReviewRole("architect", "架构审查"),
            WorktreeReviewRole("tester", "测试审查"),
            WorktreeReviewRole("integration", "集成审查"),
            WorktreeReviewRole("product", "目标验收"),
        ]
        haystack = " ".join([goal, *changed_files]).lower()
        if any(
            token in haystack
            for token in ("auth", "token", "secret", "permission", "security")
        ):
            roles.append(WorktreeReviewRole("security", "安全审查"))
        if any(
            path.endswith((".md", ".rst")) or "/docs/" in path
            for path in changed_files
        ):
            roles.append(WorktreeReviewRole("docs", "文档审查", blocking=False))
        return WorktreeReviewPlan(roles=roles)

    @staticmethod
    def _build_prompt(
        *,
        goal: str,
        unit: WorktreeUnit,
        diff: str,
        touched_files: list[str],
        input_tests: list[str],
        input_findings: list[str],
    ) -> str:
        evidence_payload = {
            "unit_id": unit.unit_id,
            "task_title": unit.task_title,
            "execution_summary": _truncate(unit.summary),
            "execution_stop_reason": unit.stop_reason,
            "touched_files": touched_files,
            "reported_tests": input_tests,
            "reported_findings": input_findings,
        }
        return f"""你是 Worktree 的独立终态评审员。必须基于给定的真实证据评审，必要时在当前 worktree 运行验证命令。

## 用户目标
{goal}

## 执行事实
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}

## 当前 Worktree 真实 Diff
```diff
{diff or "(no diff captured)"}
```

只输出严格 JSON 对象：
{{
  "verdict": "PASS|FAIL|INCONCLUSIVE",
  "summary": "结论摘要",
  "tests": [
    {{"command": "实际运行的命令", "passed": true, "evidence": "输出证据"}}
  ],
  "findings": [
    {{"severity": "blocker|major|minor|observation", "message": "发现", "evidence": "diff/测试证据", "target": "文件"}}
  ]
}}

规则：PASS 必须至少有一条带 evidence 的 finding，且至少有一条实际测试记录且全部通过；空 findings 不得 PASS。
"""

    @staticmethod
    def _outcome_from_payload(
        payload: dict[str, Any],
        *,
        evidence: dict[str, Any],
    ) -> WorktreeReviewOutcome:
        findings = _normalize_findings(payload.get("findings"))
        tests = _normalize_tests(payload.get("tests"))
        blockers = [
            finding
            for finding in findings
            if finding["severity"] in {"blocker", "major"}
        ]
        observations = [finding for finding in findings if finding not in blockers]
        requested = str(payload.get("verdict") or "").strip().upper()
        error_code = ""
        test_executions = evidence.get("test_executions")
        if not isinstance(test_executions, list):
            test_executions = []
        has_unverified, has_unknown_exit, has_failed_execution = (
            _bind_test_provenance(tests, test_executions)
        )

        if requested == WorktreeReviewVerdict.FAIL.value or blockers:
            verdict = WorktreeReviewVerdict.FAIL
            error_code = "review_failed"
        elif not findings:
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "empty_findings"
        elif not any(str(item.get("evidence") or "").strip() for item in findings):
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "findings_without_evidence"
        elif not tests:
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "missing_test_evidence"
        elif any(
            not str(test.get("command") or "").strip()
            or not str(test.get("evidence") or "").strip()
            for test in tests
        ):
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "invalid_test_evidence"
        elif any(test.get("passed") is not True for test in tests):
            verdict = WorktreeReviewVerdict.FAIL
            error_code = "tests_failed"
        elif has_failed_execution:
            verdict = WorktreeReviewVerdict.FAIL
            error_code = "tests_failed"
        elif has_unknown_exit:
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "unknown_test_exit_code"
        elif has_unverified:
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "unverified_test_evidence"
        elif requested == WorktreeReviewVerdict.PASS.value:
            verdict = WorktreeReviewVerdict.PASS
        else:
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
            error_code = "invalid_verdict"

        evidence = {
            **evidence,
            "output_finding_count": len(findings),
            "output_test_count": len(tests),
        }
        return WorktreeReviewOutcome(
            verdict=verdict,
            summary=str(payload.get("summary") or "").strip(),
            findings=findings,
            tests=tests,
            blockers=blockers,
            observations=observations,
            evidence=evidence,
            error_code=error_code,
        )

    def review_unit(
        self,
        *,
        goal: str,
        unit: WorktreeUnit,
        timeout: float = 240.0,
        base_branch: str | None = None,
    ) -> WorktreeReviewOutcome:
        diff, touched_files = _capture_git_evidence(
            unit.worktree_path,
            base_branch=base_branch,
        )
        input_tests: list[str] = []
        for key in ("test_results", "tests", "verification"):
            input_tests.extend(_metadata_lines(unit.metadata.get(key)))
        dispatcher_executions = _dispatcher_execution_evidence(
            unit.metadata.get("test_provenance")
        )
        input_tests.extend(_metadata_lines(dispatcher_executions))
        input_findings = _metadata_lines(unit.metadata.get("findings"))
        if unit.summary:
            input_findings.append(_truncate(unit.summary))
        evidence = {
            "unit_id": unit.unit_id,
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest()
            if diff
            else "",
            "diff_chars": len(diff),
            "touched_files": touched_files,
            "input_test_count": len(input_tests),
            "dispatcher_execution_count": len(dispatcher_executions),
            "input_finding_count": len(input_findings),
        }
        if unit.has_changes and not diff:
            return WorktreeReviewOutcome(
                verdict=WorktreeReviewVerdict.INCONCLUSIVE,
                summary="单元声明存在变更，但未能采集真实 diff",
                evidence=evidence,
                error_code="missing_diff",
            )

        prompt = self._build_prompt(
            goal=goal,
            unit=unit,
            diff=diff,
            touched_files=touched_files,
            input_tests=input_tests,
            input_findings=input_findings,
        )
        session = None
        try:
            session = self._session_factory(
                agent_type=self._agent_type(unit),
                cwd=unit.worktree_path,
                model_name=unit.model_name,
            )
            result = session.send_prompt(prompt, timeout=timeout)
            assessment = classify_prompt_result(result)
            if assessment.outcome is not PromptOutcome.COMPLETED:
                return WorktreeReviewOutcome(
                    verdict=WorktreeReviewVerdict.INCONCLUSIVE,
                    summary="评审会话未正常完成",
                    evidence=evidence,
                    error_code=f"review_{assessment.stop_reason}",
                )
            review_executions = _review_execution_evidence(result)
            evidence["review_execution_count"] = len(review_executions)
            evidence["review_executions"] = review_executions
            evidence["test_executions"] = [
                *dispatcher_executions,
                *review_executions,
            ]
            payload = _extract_json_object(str(getattr(result, "text", result) or ""))
            outcome = self._outcome_from_payload(payload, evidence=evidence)
            post_diff, _post_files = _capture_git_evidence(
                unit.worktree_path,
                base_branch=base_branch,
            )
            if hashlib.sha256(post_diff.encode("utf-8")).digest() != hashlib.sha256(
                diff.encode("utf-8")
            ).digest():
                outcome.verdict = WorktreeReviewVerdict.FAIL
                outcome.error_code = "review_mutated_worktree"
                outcome.summary = "独立评审修改了 Worktree，已拒绝终态和合并"
            return outcome
        except Exception as exc:
            return WorktreeReviewOutcome(
                verdict=WorktreeReviewVerdict.INCONCLUSIVE,
                summary=f"评审证据无法确认: {get_error_detail(exc)}",
                evidence=evidence,
                error_code="review_error",
            )
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    logger.debug("[WorktreeReview] close failed", exc_info=True)

    def review_units(
        self,
        *,
        goal: str,
        units: Iterable[WorktreeUnit],
        timeout: float = 240.0,
        base_branch: str | None = None,
    ) -> WorktreeReviewOutcome:
        unit_list = list(units)
        if not unit_list:
            return WorktreeReviewOutcome(error_code="no_units")
        outcomes = [
            self.review_unit(
                goal=goal,
                unit=unit,
                timeout=timeout,
                base_branch=base_branch,
            )
            for unit in unit_list
        ]
        findings = [finding for outcome in outcomes for finding in outcome.findings]
        tests = [test for outcome in outcomes for test in outcome.tests]
        blockers = [finding for outcome in outcomes for finding in outcome.blockers]
        observations = [
            finding for outcome in outcomes for finding in outcome.observations
        ]
        if any(outcome.verdict is WorktreeReviewVerdict.FAIL for outcome in outcomes):
            verdict = WorktreeReviewVerdict.FAIL
        elif any(
            outcome.verdict is WorktreeReviewVerdict.INCONCLUSIVE
            for outcome in outcomes
        ):
            verdict = WorktreeReviewVerdict.INCONCLUSIVE
        else:
            verdict = WorktreeReviewVerdict.PASS
        first_error = next(
            (outcome.error_code for outcome in outcomes if outcome.error_code),
            "",
        )
        return WorktreeReviewOutcome(
            verdict=verdict,
            summary="；".join(outcome.summary for outcome in outcomes if outcome.summary),
            findings=findings,
            tests=tests,
            blockers=blockers,
            observations=observations,
            evidence={
                "unit_count": len(unit_list),
                "reviewed_unit_count": len(outcomes),
            },
            error_code=first_error,
            unit_outcomes=[outcome.to_dict() for outcome in outcomes],
        )

    def aggregate(self, findings: list[dict[str, Any]]) -> WorktreeReviewOutcome:
        """Compatibility aggregator that now fails closed on empty evidence."""
        payload = {
            "verdict": "PASS" if findings else "INCONCLUSIVE",
            "summary": "兼容 findings 聚合",
            "tests": [],
            "findings": findings,
        }
        return self._outcome_from_payload(
            payload,
            evidence={"input_finding_count": len(findings)},
        )
