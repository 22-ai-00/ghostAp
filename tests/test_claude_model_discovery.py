from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.acp import helper


@pytest.mark.parametrize(
    ("current_model", "expected_models"),
    [
        (None, []),
        ("claude-opus-4-8[1m]", [("claude-opus-4-8[1m]", True)]),
    ],
)
def test_claude_cli_model_discovery_never_starts_an_acp_server_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    current_model: str | None,
    expected_models: list[tuple[str, bool]],
) -> None:
    """Claude's default is represented by no explicit model override.

    A saved explicit model remains selectable, but neither case may ask the
    CLI-only backend for the unsupported ``claude acp serve`` transport.
    """
    get_serve_command = MagicMock(
        side_effect=RuntimeError("Claude CLI has no ACP server mode")
    )
    provider = SimpleNamespace(get_serve_command=get_serve_command)
    monkeypatch.setattr(
        helper,
        "get_providers",
        lambda: {"claude": provider},
    )

    probe_async = AsyncMock(wraps=helper._probe_acp_models)
    probe_blocking = MagicMock(wraps=helper._probe_blocking)
    monkeypatch.setattr(helper, "_probe_acp_models", probe_async)
    monkeypatch.setattr(helper, "_probe_blocking", probe_blocking)

    models = helper.fetch_acp_models(
        "claude",
        str(tmp_path),
        current_model=current_model,
    )

    assert [(model.name, model.is_default) for model in models] == expected_models
    assert (
        probe_blocking.call_count,
        probe_async.await_count,
        get_serve_command.call_count,
    ) == (0, 0, 0)

