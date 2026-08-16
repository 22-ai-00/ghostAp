from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

from src.autonomous.remote.models import (
    A2A_SPEC_RELEASE,
    A2A_WIRE_PROTOCOL_VERSION,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_a2a_dependencies_are_direct_and_exactly_pinned() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())

    assert "a2a-sdk==1.1.2" in pyproject["project"]["dependencies"]
    assert "httpx==0.28.1" in pyproject["project"]["dependencies"]
    assert importlib.metadata.version("a2a-sdk") == "1.1.2"
    assert importlib.metadata.version("httpx") == "0.28.1"


def test_a2a_spec_and_wire_versions_are_independently_frozen() -> None:
    assert A2A_SPEC_RELEASE == "1.0.1"
    assert A2A_WIRE_PROTOCOL_VERSION == "1.0"
