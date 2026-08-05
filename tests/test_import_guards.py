"""Verify that public __all__ symbols in key modules are importable."""

import importlib
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


def test_domain_compat_entries_exist_for_spec_and_ttadk_utils() -> None:
    """Task 28 guard: domain packages expose compatibility entries while old utils stay importable."""
    spec_utils = importlib.import_module("src.spec_engine.utils")
    legacy_spec_utils = importlib.import_module("src.utils.spec_utils")
    ttadk_wrapper = importlib.import_module("src.ttadk.wrapper")
    legacy_ttadk_wrapper = importlib.import_module("src.utils.ttadk_wrapper")

    assert spec_utils.extract_json_blob is legacy_spec_utils.extract_json_blob
    assert spec_utils.parse_review_output_loose is legacy_spec_utils.parse_review_output_loose
    assert ttadk_wrapper.WrapperState is legacy_ttadk_wrapper.WrapperState
    assert ttadk_wrapper.pump_filtered_stream is legacy_ttadk_wrapper.pump_filtered_stream


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


def test_production_code_does_not_call_cardevent_worktree_compat_shims() -> None:
    """Refactoring-analysis guard: production paths import src.card.events.worktree factories directly."""
    root = Path(__file__).parent.parent / "src"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.parts[-3:] == ("card", "events", "factories.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if "CardEvent.worktree_" in text:
            offenders.append(str(path.relative_to(root.parent)))

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
