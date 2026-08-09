"""AC-001: prevent retired Slock runtime surfaces from returning to ``src``."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
RETIRED_REFERENCE = re.compile(r"slock", re.IGNORECASE)
RETIRED_SYMBOLS = {
    "SlockCommandAction",
    "SlockEngineCallbacks",
    "SlockHandler",
    "TeamChannel",
    "TeamEngine",
    "TeamEngineManager",
}
RETIRED_ROUTE_TARGETS = {
    "/council",
    "/discuss",
    "/new-role",
    "/new-team",
    "/plan",
    "/role",
    "/slock",
    "/slocks",
    "/team",
}
RETIRED_ACTION_PREFIXES = ("slock_",)
CACHE_DIRECTORY_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
ABI_LITERAL_PATHS = {
    Path("config/settings.py"),
    Path("autonomous/workforce/identity.py"),
}
ABI_LITERALS = {
    "~/.ghostap/slock",
    "~/.ghostap/slock/credentials",
}


def _is_effective_source_path(path: Path) -> bool:
    return not any(part in CACHE_DIRECTORY_NAMES for part in path.parts)


def _source_files(suffix: str) -> Iterator[Path]:
    yield from (
        path
        for path in SOURCE_ROOT.rglob(f"*{suffix}")
        if _is_effective_source_path(path)
    )


def _relative_path(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _has_retired_reference(value: str) -> bool:
    stripped = value.strip()
    return (
        RETIRED_REFERENCE.search(value) is not None
        or stripped in RETIRED_SYMBOLS
        or stripped in RETIRED_ROUTE_TARGETS
        or any(stripped.startswith(prefix) for prefix in RETIRED_ACTION_PREFIXES)
    )


def _is_approved_abi_literal(path: Path, value: str) -> bool:
    return path.relative_to(SOURCE_ROOT) in ABI_LITERAL_PATHS and value in ABI_LITERALS


def _python_location(path: Path, node: ast.AST) -> str:
    return f"{_relative_path(path)}:{node.lineno}:{node.col_offset + 1}"


def _python_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    if "\0" in source:
        return []
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    relative_to_source = path.relative_to(SOURCE_ROOT)

    for part in relative_to_source.parts:
        if _has_retired_reference(part):
            violations.append(
                f"{_relative_path(path)} [module path] contains retired reference: {part!r}"
            )
            break

    for node in ast.walk(tree):
        label: str | None = None
        value: str | None = None
        if isinstance(node, ast.ImportFrom) and node.module and _has_retired_reference(node.module):
            violations.append(
                f"{_python_location(path, node)} [import module] "
                f"contains retired reference: {node.module!r}"
            )
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            label, value = node.__class__.__name__, node.name
        elif isinstance(node, ast.Name):
            label, value = "identifier", node.id
        elif isinstance(node, ast.Attribute):
            label, value = "attribute", node.attr
        elif isinstance(node, ast.arg):
            label, value = "argument", node.arg
        elif isinstance(node, ast.alias):
            label, value = "import", node.name
            if node.asname and _has_retired_reference(node.asname):
                violations.append(
                    f"{_python_location(path, node)} [import alias] "
                    f"contains retired reference: {node.asname!r}"
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            label, value = "runtime string", node.value

        if value is None or not _has_retired_reference(value):
            continue
        if label == "runtime string" and _is_approved_abi_literal(path, value):
            continue
        violations.append(
            f"{_python_location(path, node)} [{label}] contains retired reference: {value!r}"
        )

    return violations


def _json_pointer(tokens: tuple[str, ...]) -> str:
    return "/" + "/".join(token.replace("~", "~0").replace("/", "~1") for token in tokens)


def _json_values(value: Any, selector: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            key_selector = (*selector, key_text)
            yield _json_pointer(key_selector), key_text
            yield from _json_values(child, key_selector)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _json_values(child, (*selector, str(index)))
    elif isinstance(value, str):
        yield _json_pointer(selector), value


def _json_violations(path: Path) -> list[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        f"{_relative_path(path)} [json selector {selector}] contains retired reference: {value!r}"
        for selector, value in _json_values(document)
        if _has_retired_reference(value)
    ]


def test_retired_runtime_reference_inventory_covers_non_slock_surfaces() -> None:
    for value in ("TeamEngine", "TeamEngineManager", "TeamChannel", "/new-team"):
        assert _has_retired_reference(value)

    for retained_employee_route in ("/task", "/memory"):
        assert not _has_retired_reference(retained_employee_route)


def test_no_retired_slock_runtime_surfaces_remain() -> None:
    """Catch imports, symbols, routes/actions/registrations/engine strings, and JSON selectors."""
    violations = [
        violation
        for path in _source_files(".py")
        for violation in _python_violations(path)
    ]
    violations.extend(
        violation
        for path in _source_files(".json")
        for violation in _json_violations(path)
    )

    assert not violations, (
        "Retired Slock runtime surfaces remain; remove every listed reference.\n"
        + "\n".join(sorted(violations))
    )
