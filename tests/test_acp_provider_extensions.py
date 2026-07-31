from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.acp.providers import (
    CODEX_ACP_NPM_PACKAGE,
    AidenProvider,
    CodexProvider,
    GeminiProvider,
    TraexProvider,
    _get_aiden_acp_serve_help_blob,
    _get_codex_acp_serve_help_blob,
    _get_gemini_acp_serve_help_blob,
    _get_traex_acp_serve_help_blob,
    _reset_providers_for_testing,
)
from src.acp.sync_adapter import resolve_agent_spec


def test_aiden_provider_name():
    assert AidenProvider().name == "aiden"


@patch("src.acp.providers.subprocess.run")
def test_aiden_provider_availability(mock_run):
    _get_aiden_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout="Usage: aiden acp [options]\nRun Aiden CLI as an ACP agent for editors like Zed\n",
        stderr="",
    )
    assert AidenProvider().check_availability() is True

    _get_aiden_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="", stderr="")
    assert AidenProvider().check_availability() is False


@patch("src.acp.providers.subprocess.run")
def test_aiden_provider_serve_command_model_style_long(mock_run):
    _get_aiden_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: aiden acp --model MODEL", stderr="")
    cmd, args = AidenProvider().get_serve_command("gpt-4")
    assert cmd == "aiden"
    assert args == ["acp", "--model", "gpt-4"]


@patch("src.acp.providers.subprocess.run")
def test_aiden_provider_serve_command_model_style_short(mock_run):
    _get_aiden_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: aiden acp -m MODEL", stderr="")
    cmd, args = AidenProvider().get_serve_command("gpt-4")
    assert cmd == "aiden"
    assert args == ["acp", "-m", "gpt-4"]


def test_codex_provider_name():
    assert CodexProvider().name == "codex"


def test_codex_fallback_package_is_official_agentclientprotocol_adapter():
    assert CODEX_ACP_NPM_PACKAGE == "@agentclientprotocol/codex-acp@1.1.7"


@patch("src.acp.providers.shutil.which")
@patch("src.acp.sync_adapter._probe_acp_serve_help")
def test_codex_provider_availability(mock_probe, mock_which):
    mock_probe.return_value = (True, 0, "Usage: codex acp serve", "")
    mock_which.return_value = None
    assert CodexProvider().check_availability() is True

    mock_probe.return_value = (False, 1, "", "")
    assert CodexProvider().check_availability() is False

    mock_which.return_value = "/usr/bin/npx"
    assert CodexProvider().check_availability() is True


@patch("src.acp.sync_adapter._probe_acp_serve_help")
def test_codex_provider_serve_command_model_style_long(mock_probe):
    _get_codex_acp_serve_help_blob.cache_clear()
    mock_probe.return_value = (True, 0, "Usage: codex acp serve --model MODEL", "")
    cmd, args = CodexProvider().get_serve_command("gpt-4")
    assert cmd == "codex"
    assert args == ["acp", "serve", "--model", "gpt-4"]


@patch("src.acp.sync_adapter._probe_acp_serve_help")
def test_codex_provider_serve_command_model_style_short(mock_probe):
    _get_codex_acp_serve_help_blob.cache_clear()
    mock_probe.return_value = (True, 0, "Usage: codex acp serve -m MODEL", "")
    cmd, args = CodexProvider().get_serve_command("gpt-4")
    assert cmd == "codex"
    assert args == ["acp", "serve", "-m", "gpt-4"]


@patch("src.acp.providers.shutil.which")
@patch("src.acp.sync_adapter._probe_acp_serve_help")
def test_codex_provider_falls_back_to_official_codex_acp_without_zed_config_arg(mock_probe, mock_which):
    mock_probe.return_value = (False, 2, "", "error: unrecognized subcommand 'serve'")
    mock_which.return_value = "/usr/bin/npx"

    cmd, args = CodexProvider().get_serve_command("gpt-5")

    assert cmd == "npx"
    assert args == ["--yes", CODEX_ACP_NPM_PACKAGE]


@patch("src.acp.providers.shutil.which")
@patch("src.acp.sync_adapter._probe_acp_serve_help")
def test_codex_provider_unavailable_without_native_or_npx(mock_probe, mock_which):
    mock_probe.return_value = (False, 2, "", "error: unrecognized subcommand 'serve'")
    mock_which.return_value = None

    assert CodexProvider().check_availability() is False
    with pytest.raises(RuntimeError, match="Codex ACP is unavailable"):
        CodexProvider().get_serve_command("gpt-5")


@patch("src.acp.providers.shutil.which")
@patch("src.acp.sync_adapter._probe_acp_serve_help")
def test_resolve_agent_spec_uses_registered_codex_provider_fallback(mock_probe, mock_which):
    _reset_providers_for_testing()
    mock_probe.return_value = (False, 2, "", "error: unrecognized subcommand 'serve'")
    mock_which.return_value = "/usr/bin/npx"

    cmd, args = resolve_agent_spec("codex", model_name="gpt-5")

    assert cmd == "npx"
    assert args == ["--yes", CODEX_ACP_NPM_PACKAGE]


def test_gemini_provider_name():
    assert GeminiProvider().name == "gemini"


@patch("src.acp.providers.subprocess.run")
def test_gemini_provider_availability(mock_run):
    _get_gemini_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: gemini [options]\n  --acp  Starts the agent in ACP mode\n", stderr="")
    assert GeminiProvider().check_availability() is True

    _get_gemini_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: gemini [options]\n", stderr="")
    assert GeminiProvider().check_availability() is False


@patch("src.acp.providers.subprocess.run")
def test_gemini_provider_serve_command_model_style_long(mock_run):
    _get_gemini_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: gemini [options]\n  --model MODEL\n  --acp\n", stderr="")
    cmd, args = GeminiProvider().get_serve_command("gemini-2.5-pro")
    assert cmd == "gemini"
    assert args == ["--acp", "--model", "gemini-2.5-pro"]


@patch("src.acp.providers.subprocess.run")
def test_gemini_provider_serve_command_model_style_short(mock_run):
    _get_gemini_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: gemini [options]\n  -m MODEL\n  --acp\n", stderr="")
    cmd, args = GeminiProvider().get_serve_command("gemini-2.5-pro")
    assert cmd == "gemini"
    assert args == ["--acp", "-m", "gemini-2.5-pro"]


def test_traex_provider_name():
    assert TraexProvider().name == "traex"


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_availability(mock_run):
    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout="Usage: traecli acp serve [OPTIONS]\nStart the ACP server over stdio\n",
        stderr="",
    )
    assert TraexProvider().check_availability() is True

    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(stdout="Usage: traecli [OPTIONS]\n", stderr="")
    assert TraexProvider().check_availability() is False


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_serve_command_uses_config_model(mock_run):
    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\nExamples: -c model="o3"\n',
        stderr="",
    )
    cmd, args = TraexProvider().get_serve_command("gpt-5")
    assert cmd == "traex"
    assert args == ["acp", "serve", "-c", 'model="gpt-5"']


@patch("src.acp.providers.subprocess.run")
def test_resolve_agent_spec_defers_traex_model_selection_to_acp_session(mock_run):
    _reset_providers_for_testing()
    mock_run.return_value = SimpleNamespace(
        stdout="Usage: traecli acp serve [OPTIONS]\nStart the ACP server over stdio\n",
        stderr="",
    )

    cmd, args = resolve_agent_spec("traex", model_name="gpt-5")

    assert cmd == "traex"
    assert args == ["acp", "serve"]


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_resolves_config_name_to_slug(mock_run, tmp_path):
    """Traex provider translates config_name → slug to avoid metadata lookup failure."""
    import json

    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\n',
        stderr="",
    )

    cache_data = {
        "models": [
            {"slug": "Test-O-New-Thinking", "config_name": "c_o_new_thinking"},
            {"slug": "GPT-5.5", "config_name": "gpt-5.5"},
        ]
    }
    cache_file = tmp_path / "models_cache.json"
    cache_file.write_text(json.dumps(cache_data))

    provider = TraexProvider()
    # Clear cached slug map so it re-reads
    provider._load_slug_map.cache_clear()

    with patch("pathlib.Path.home", return_value=tmp_path.parent):
        # Arrange: put cache at <home>/.trae/cli/models_cache.json
        trae_dir = tmp_path.parent / ".trae" / "cli"
        trae_dir.mkdir(parents=True, exist_ok=True)
        (trae_dir / "models_cache.json").write_text(json.dumps(cache_data))

        provider._load_slug_map.cache_clear()
        cmd, args = provider.get_serve_command("c_o_new_thinking")

    assert cmd == "traex"
    assert args == ["acp", "serve", "-c", 'model="Test-O-New-Thinking"']


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_passes_through_unknown_model(mock_run):
    """Models not in slug map are passed through unchanged."""
    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\n',
        stderr="",
    )

    provider = TraexProvider()
    provider._load_slug_map.cache_clear()

    cmd, args = provider.get_serve_command("some-unknown-model")
    assert cmd == "traex"
    assert args == ["acp", "serve", "-c", 'model="some-unknown-model"']


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_strips_compound_variant_suffix_before_slug(mock_run, tmp_path):
    """Compound cascade values (config_name/profile/effort) resolve via base slug.

    Regression for Deep+Traex Internal error: a value like
    "c_o_new_thinking/max/max" was passed verbatim to `-c model=`, causing the
    Traex CLI to fail metadata lookup and return Internal error.
    """
    import json

    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\n',
        stderr="",
    )

    cache_data = {
        "models": [
            {"slug": "Test-O-New-Thinking", "config_name": "c_o_new_thinking"},
        ]
    }

    provider = TraexProvider()
    provider._load_slug_map.cache_clear()

    with patch("pathlib.Path.home", return_value=tmp_path.parent):
        trae_dir = tmp_path.parent / ".trae" / "cli"
        trae_dir.mkdir(parents=True, exist_ok=True)
        (trae_dir / "models_cache.json").write_text(json.dumps(cache_data))

        provider._load_slug_map.cache_clear()
        cmd, args = provider.get_serve_command("c_o_new_thinking/max/max")

    assert cmd == "traex"
    # Base config_name resolves to slug; compound suffix is stripped.
    assert args == ["acp", "serve", "-c", 'model="Test-O-New-Thinking"']


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_strips_standard_profile_before_slug(mock_run, tmp_path):
    """The explicit standard profile is syntax, not part of the model slug."""
    import json

    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\n',
        stderr="",
    )
    cache_data = {
        "models": [
            {"slug": "Test-O-New-Thinking", "config_name": "c_o_new_thinking"},
        ]
    }
    provider = TraexProvider()
    provider._load_slug_map.cache_clear()

    with patch("pathlib.Path.home", return_value=tmp_path.parent):
        trae_dir = tmp_path.parent / ".trae" / "cli"
        trae_dir.mkdir(parents=True, exist_ok=True)
        (trae_dir / "models_cache.json").write_text(json.dumps(cache_data))
        provider._load_slug_map.cache_clear()
        cmd, args = provider.get_serve_command(
            "c_o_new_thinking/standard/high"
        )

    assert cmd == "traex"
    assert args == ["acp", "serve", "-c", 'model="Test-O-New-Thinking"']


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_compound_unknown_model_falls_back_to_base(mock_run):
    """Unknown compound value falls back to base config_name (never the compound)."""
    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\n',
        stderr="",
    )

    provider = TraexProvider()
    provider._load_slug_map.cache_clear()

    cmd, args = provider.get_serve_command("openrouter-3o/max/high")
    assert cmd == "traex"
    assert args == ["acp", "serve", "-c", 'model="openrouter-3o"']


@patch("src.acp.providers.subprocess.run")
def test_traex_provider_preserves_ordinary_slash_model(mock_run):
    """Ordinary slash-bearing names without variant tokens are preserved intact."""
    _get_traex_acp_serve_help_blob.cache_clear()
    mock_run.return_value = SimpleNamespace(
        stdout='Usage: traecli acp serve [OPTIONS]\n  -c, --config <key=value>\n',
        stderr="",
    )

    provider = TraexProvider()
    provider._load_slug_map.cache_clear()

    cmd, args = provider.get_serve_command("anthropic/claude-sonnet")
    assert cmd == "traex"
    assert args == ["acp", "serve", "-c", 'model="anthropic/claude-sonnet"']
