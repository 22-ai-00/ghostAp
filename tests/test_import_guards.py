"""Verify that public __all__ symbols in key modules are importable."""

import re
from pathlib import Path


def test_card_builder_project_context_import_is_type_checking_only() -> None:
    """Task 27 guard: card builder must not runtime-import project context for annotations."""
    source = (Path(__file__).parent.parent / "src" / "card" / "builder.py").read_text(encoding="utf-8")

    assert "from __future__ import annotations" in source
    assert "from ..project.context import ProjectContext" not in source.split("if TYPE_CHECKING:", 1)[0]


def test_card_builder_implementations_do_not_runtime_import_project_context() -> None:
    root = Path(__file__).parent.parent
    offenders: list[str] = []
    for path in (root / "src" / "card" / "builders").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        runtime_section = source.split("if TYPE_CHECKING:", 1)[0]
        if "ProjectContext" in runtime_section and "import ProjectContext" in runtime_section:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []




def test_card_styles_compat_module_removed_and_not_referenced_by_production() -> None:
    """Refactoring-analysis guard: legacy styles.py re-export module stays removed."""
    root = Path(__file__).parent.parent
    assert not (root / "src" / "card" / "styles.py").exists()

    offenders: list[str] = []
    for path in (root / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"(?:from|import)\s+src\.card\.styles\b|from\s+\.{1,3}styles\s+import\b", source):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_removed_worktree_runtime_modules_are_not_imported() -> None:
    """Retired Worktree modules must not remain reachable from production imports."""
    root = Path(__file__).parent.parent
    offenders: list[str] = []
    removed_import_fragments = (
        "worktree_engine",
        "handlers.worktree",
        "renderers.worktree_renderer",
    )
    for path in (root / "src").rglob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not re.match(r"^\s*(?:from|import)\s+", line):
                continue
            if any(fragment in line for fragment in removed_import_fragments):
                offenders.append(f"{path.relative_to(root)}:{line_number}")

    assert offenders == []


def test_error_diagnostics_do_not_live_in_generic_utils() -> None:
    """Feishu/card diagnostic security state must not be a process bearer token in utils."""
    root = Path(__file__).parent.parent

    assert not (root / "src" / "utils" / "error_diagnostics.py").exists()

    offenders: list[str] = []
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "utils.error_diagnostics" in text or "src.utils.error_diagnostics" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
