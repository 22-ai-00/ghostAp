from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.acp import helper


@pytest.mark.parametrize(
    ("current_model", "expected_models"),
    [
        # No saved model: the synthetic "default" pseudo-base is active and
        # carries the effort dimension; it never resolves to a --model flag.
        (None, [("default", True)]),
        # A saved composite round-trips as an extra base flagged default,
        # ahead of the still-present "let the gateway pick" pseudo-base.
        (
            "claude-opus-4-8[1m]",
            [("default", False), ("claude-opus-4-8[1m]", True)],
        ),
    ],
)
def test_claude_cli_model_discovery_never_starts_an_acp_server_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    current_model: str | None,
    expected_models: list[tuple[str, bool]],
) -> None:
    """Claude CLI discovery is a purely synthetic base[1m]/effort matrix.

    The catalog is built without probing, so neither a fresh nor a saved
    selection may ask the CLI-only backend for the unsupported
    ``claude acp serve`` transport.
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

