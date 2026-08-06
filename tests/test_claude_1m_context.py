"""Unit tests for Claude 1M-context support.

Covers the small surface introduced for the ``[1m]`` model-id suffix:
capabilities helpers, ANTHROPIC_BETAS env injection, the probe-side
variant injector, ``_apply_model_args`` round-tripping the suffix, and
the provider config flag that finally makes ``--model`` reach the CLI.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.acp.claude_capabilities import (
    CONTEXT_1M_BETA,
    SUFFIX_1M,
    is_1m_variant,
    model_supports_1m,
    strip_1m_suffix,
    with_1m_suffix,
)
from src.acp.helper import _inject_claude_1m_variants
from src.acp.options import ACPModelOption
from src.acp.providers import (
    ClaudeProvider,
    _apply_model_args,
    _reset_providers_for_testing,
    get_providers,
)
from src.utils.env import apply_anthropic_betas


def test_claude_1m_selection_reaches_real_cli_environment():
    from src.agent_session.claude_cli import ClaudeCLIConfig, SyncClaudeCLISession

    process = MagicMock()
    process.stdout = iter([])
    process.stderr = MagicMock()
    process.stderr.read.return_value = ""
    process.returncode = 0
    process.poll.return_value = 0
    process.wait.return_value = None
    process.pid = 1234
    session = SyncClaudeCLISession(
        cwd="/tmp",
        model_name="claude-opus-4-8[1m]",
        config=ClaudeCLIConfig(add_dir=False, bypass_permissions=False),
    )
    session.session_id = "session-1"

    with (
        patch(
            "src.agent_session.claude_cli.subprocess.Popen",
            return_value=process,
        ) as popen,
        patch(
            "src.utils.env.build_clean_env",
            return_value={"ANTHROPIC_BETAS": "existing-beta"},
        ),
    ):
        result = session.send_prompt("use the long context")

    assert result.stop_reason == "end_turn"
    argv = popen.call_args.args[0]
    env = popen.call_args.kwargs["env"]
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
    assert env["ANTHROPIC_BETAS"] == f"existing-beta,{CONTEXT_1M_BETA}"


# ---------------------------------------------------------------------------
# capabilities helpers
# ---------------------------------------------------------------------------
class TestCapabilities(unittest.TestCase):
    def test_supports_1m_known_prefixes(self):
        self.assertTrue(model_supports_1m("claude-sonnet-4-5"))
        self.assertTrue(model_supports_1m("claude-sonnet-4"))
        self.assertTrue(model_supports_1m("claude-opus-4-8"))
        self.assertTrue(model_supports_1m("claude-opus-4-5"))

    def test_supports_1m_with_date_stamp(self):
        # Date-stamped releases match by prefix.
        self.assertTrue(model_supports_1m("claude-opus-4-8-20260101"))
        self.assertTrue(model_supports_1m("claude-sonnet-4-5-20260101"))

    def test_supports_1m_negatives(self):
        self.assertFalse(model_supports_1m("claude-haiku-4-5"))
        self.assertFalse(model_supports_1m("gpt-5.2"))
        self.assertFalse(model_supports_1m(""))
        self.assertFalse(model_supports_1m("   "))

    def test_supports_1m_strips_suffix_before_match(self):
        self.assertTrue(model_supports_1m("claude-opus-4-8[1m]"))

    def test_is_1m_variant(self):
        self.assertTrue(is_1m_variant("claude-opus-4-8[1m]"))
        self.assertFalse(is_1m_variant("claude-opus-4-8"))
        self.assertFalse(is_1m_variant(""))

    def test_strip_1m_suffix(self):
        self.assertEqual(strip_1m_suffix("claude-opus-4-8[1m]"), "claude-opus-4-8")
        # Idempotent on inputs that don't carry the suffix.
        self.assertEqual(strip_1m_suffix("claude-opus-4-8"), "claude-opus-4-8")
        self.assertEqual(strip_1m_suffix(""), "")

    def test_with_1m_suffix(self):
        self.assertEqual(with_1m_suffix("claude-opus-4-8"), "claude-opus-4-8[1m]")
        # Idempotent.
        self.assertEqual(
            with_1m_suffix("claude-opus-4-8[1m]"), "claude-opus-4-8[1m]"
        )

    def test_constants(self):
        self.assertEqual(SUFFIX_1M, "[1m]")
        self.assertEqual(CONTEXT_1M_BETA, "context-1m-2025-08-07")


# ---------------------------------------------------------------------------
# apply_anthropic_betas
# ---------------------------------------------------------------------------
class TestApplyAnthropicBetas(unittest.TestCase):
    def test_noop_for_non_1m_model(self):
        env: dict[str, str] = {"FOO": "bar"}
        out = apply_anthropic_betas(env, "claude-haiku-4-5")
        self.assertIs(out, env)
        self.assertNotIn("ANTHROPIC_BETAS", env)

    def test_noop_for_empty_model(self):
        env: dict[str, str] = {}
        apply_anthropic_betas(env, None)
        apply_anthropic_betas(env, "")
        self.assertNotIn("ANTHROPIC_BETAS", env)

    def test_sets_beta_for_1m_variant(self):
        env: dict[str, str] = {}
        out = apply_anthropic_betas(env, "claude-opus-4-8[1m]")
        self.assertIs(out, env)
        self.assertEqual(env["ANTHROPIC_BETAS"], CONTEXT_1M_BETA)

    def test_merges_with_existing_betas(self):
        env = {"ANTHROPIC_BETAS": "other-beta-1,another-beta"}
        apply_anthropic_betas(env, "claude-opus-4-8[1m]")
        self.assertEqual(
            env["ANTHROPIC_BETAS"],
            f"other-beta-1,another-beta,{CONTEXT_1M_BETA}",
        )

    def test_dedups_existing_beta(self):
        env = {"ANTHROPIC_BETAS": f"  {CONTEXT_1M_BETA} , other-beta "}
        apply_anthropic_betas(env, "claude-opus-4-8[1m]")
        # Already present → not appended again; existing value normalised.
        self.assertEqual(
            env["ANTHROPIC_BETAS"], f"{CONTEXT_1M_BETA},other-beta"
        )

    def test_does_not_touch_betas_for_non_1m_even_if_set(self):
        env = {"ANTHROPIC_BETAS": "preserved-beta"}
        apply_anthropic_betas(env, "claude-opus-4-8")
        self.assertEqual(env["ANTHROPIC_BETAS"], "preserved-beta")


# ---------------------------------------------------------------------------
# _apply_model_args preserves the [1m] suffix
# ---------------------------------------------------------------------------
class TestApplyModelArgsPreserves1mSuffix(unittest.TestCase):
    def test_model_long_keeps_suffix(self):
        out = _apply_model_args([], "claude-opus-4-8[1m]", "model_long", None)
        self.assertEqual(out, ["--model", "claude-opus-4-8[1m]"])

    def test_model_long_plain(self):
        out = _apply_model_args([], "claude-opus-4-8", "model_long", None)
        self.assertEqual(out, ["--model", "claude-opus-4-8"])


# ---------------------------------------------------------------------------
# Provider config: claude must use model_long
# ---------------------------------------------------------------------------
class TestClaudeProviderConfig(unittest.TestCase):
    def setUp(self):
        _reset_providers_for_testing()

    def tearDown(self):
        _reset_providers_for_testing()

    def test_claude_model_style_is_model_long(self):
        providers = get_providers()
        self.assertIn("claude", providers)
        self.assertEqual(providers["claude"]._config.model_style, "model_long")

    def test_claude_serve_command_passes_model_with_suffix(self):
        # Patch the availability checker so get_serve_command works without
        # an actual `claude` binary.
        with patch.object(
            ClaudeProvider(),
            "check_availability",
            return_value=True,
        ):
            cmd, args = ClaudeProvider().get_serve_command("claude-opus-4-8[1m]")
        self.assertEqual(cmd, "claude")
        self.assertEqual(args, ["acp", "serve", "--model", "claude-opus-4-8[1m]"])


# ---------------------------------------------------------------------------
# _inject_claude_1m_variants
# ---------------------------------------------------------------------------
class TestInject1mVariants(unittest.TestCase):
    def test_appends_variant_for_supported_model(self):
        items = [
            ACPModelOption(name="claude-opus-4-8", description="Opus", is_default=True),
            ACPModelOption(name="claude-haiku-4-5", description="Haiku"),
        ]
        out = _inject_claude_1m_variants(items)
        names = [m.name for m in out]
        # Opus picks up a [1m] sibling; Haiku does not.
        self.assertIn("claude-opus-4-8", names)
        self.assertIn("claude-opus-4-8[1m]", names)
        self.assertIn("claude-haiku-4-5", names)
        self.assertNotIn("claude-haiku-4-5[1m]", names)

    def test_variant_is_marked_supports_1m_and_not_default(self):
        items = [
            ACPModelOption(name="claude-opus-4-8", description="Opus", is_default=True),
        ]
        out = _inject_claude_1m_variants(items)
        variant = next(m for m in out if m.name == "claude-opus-4-8[1m]")
        self.assertTrue(variant.supports_1m)
        self.assertFalse(variant.is_default)
        # Description carries the visual badge so the card shows it without
        # any builder-side change.
        self.assertIn("🚀", variant.description)
        self.assertIn("1M", variant.description)

    def test_idempotent_on_already_suffixed_input(self):
        items = [ACPModelOption(name="claude-opus-4-8[1m]", description="Opus 1M")]
        out = _inject_claude_1m_variants(items)
        self.assertEqual([m.name for m in out], ["claude-opus-4-8[1m]"])

    def test_does_not_double_inject_when_both_already_present(self):
        items = [
            ACPModelOption(name="claude-opus-4-8", description="Opus"),
            ACPModelOption(name="claude-opus-4-8[1m]", description="Opus 1M"),
        ]
        out = _inject_claude_1m_variants(items)
        self.assertEqual(
            sorted(m.name for m in out),
            ["claude-opus-4-8", "claude-opus-4-8[1m]"],
        )

    def test_no_changes_when_no_supported_models(self):
        items = [ACPModelOption(name="claude-haiku-4-5", description="Haiku")]
        out = _inject_claude_1m_variants(items)
        self.assertEqual([m.name for m in out], ["claude-haiku-4-5"])


if __name__ == "__main__":
    unittest.main()
