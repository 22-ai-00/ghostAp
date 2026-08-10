"""ReviewArtifacts: structured collection of cycle outputs passed to review.

Decouples the review pipeline from the live ACP session: review strategies
consume artifacts (diff / phase outputs / requirement) rather than inheriting
the build session's in-memory context. This is a necessary precondition for
the review refactor (independent review session, heterogeneous agents,
retroactive re-review).

Step 1 of the review refactor — this module only collects and serializes.
Wiring into engine.conduct_review happens in Step 7.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

from .artifacts import truncate_text

if TYPE_CHECKING:
    from .models import SpecCycle, SpecProject

logger = logging.getLogger(__name__)

__all__ = [
    "ReviewArtifacts",
    "collect_review_artifacts",
]


@dataclass
class ReviewArtifacts:
    """Snapshot of everything a reviewer needs, with zero session dependency."""

    cycle_number: int
    requirement: str
    cwd: str
    spec_output: str = ""
    plan_output: str = ""
    tasks_output: str = ""
    build_output: str = ""
    diff_patch: str = ""
    touched_files: list[str] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)

    spec_path: Optional[str] = None
    plan_path: Optional[str] = None
    tasks_path: Optional[str] = None
    build_path: Optional[str] = None

    # Completion control context (Phase 2)
    acceptance_criteria: list[str] = field(default_factory=list)
    criteria_satisfied: dict[int, bool] = field(default_factory=dict)
    verify_command: str = ""
    verify_passed: Optional[bool] = None
    verify_output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewArtifacts":
        return cls(
            cycle_number=int(data.get("cycle_number") or 0),
            requirement=str(data.get("requirement") or ""),
            cwd=str(data.get("cwd") or ""),
            spec_output=str(data.get("spec_output") or ""),
            plan_output=str(data.get("plan_output") or ""),
            tasks_output=str(data.get("tasks_output") or ""),
            build_output=str(data.get("build_output") or ""),
            diff_patch=str(data.get("diff_patch") or ""),
            touched_files=list(data.get("touched_files") or []),
            captured_at=float(data.get("captured_at") or time.time()),
            spec_path=data.get("spec_path"),
            plan_path=data.get("plan_path"),
            tasks_path=data.get("tasks_path"),
            build_path=data.get("build_path"),
            acceptance_criteria=list(data.get("acceptance_criteria") or []),
            criteria_satisfied={int(k): v for k, v in (data.get("criteria_satisfied") or {}).items()},
            verify_command=str(data.get("verify_command") or ""),
            verify_passed=data.get("verify_passed"),
            verify_output=str(data.get("verify_output") or ""),
        )


def _git_diff(cwd: str, max_bytes: int = 200_000) -> str:
    """Capture uncommitted diff. Safe-degrade on non-git dirs."""
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = proc.stdout or ""
        if len(out) > max_bytes:
            out = out[:max_bytes] + f"\n...[truncated {len(out) - max_bytes} bytes]"
        return out
    except Exception as e:
        logger.debug("[ReviewArtifacts] git diff failed: %s", repr(e))
        return ""


def _git_touched_files(cwd: str, limit: int = 100) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "diff", "HEAD", "--name-only"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        return files[:limit]
    except Exception:
        return []


def collect_review_artifacts(
    *,
    cycle: "SpecCycle",
    project: "SpecProject",
    cwd: str,
    build_output_max: int = 20_000,
    include_diff: bool = True,
    verify_passed: Optional[bool] = None,
    verify_output: str = "",
) -> ReviewArtifacts:
    """Build a ReviewArtifacts snapshot from the current cycle state.

    All text fields are truncated to reasonable limits so review prompts stay
    small regardless of build output size.
    """
    diff = _git_diff(cwd) if include_diff else ""
    files = _git_touched_files(cwd) if include_diff else []

    criteria = list(getattr(project, "acceptance_criteria", []) or [])
    tracker = getattr(project, "criteria_tracker", None)
    criteria_satisfied: dict[int, bool] = {}
    if tracker:
        criteria_satisfied = dict(getattr(tracker, "satisfied", {}) or {})

    return ReviewArtifacts(
        cycle_number=int(cycle.cycle_number),
        requirement=str(getattr(project, "requirement", "") or ""),
        cwd=cwd,
        spec_output=truncate_text(cycle.spec_content or "", 8_000),
        plan_output=truncate_text(cycle.plan_content or "", 12_000),
        tasks_output=truncate_text(
            "\n".join(t.title for t in (cycle.tasks or []) if getattr(t, "title", None)),
            4_000,
        ),
        build_output=truncate_text(cycle.build_output or "", build_output_max),
        diff_patch=diff,
        touched_files=files,
        spec_path=cycle.spec_path,
        plan_path=cycle.plan_path,
        tasks_path=cycle.tasks_path,
        build_path=cycle.build_path,
        acceptance_criteria=criteria,
        criteria_satisfied=criteria_satisfied,
        verify_command=str(getattr(project, "verify_command", "") or ""),
        verify_passed=verify_passed,
        verify_output=truncate_text(verify_output or "", 4_000),
    )
