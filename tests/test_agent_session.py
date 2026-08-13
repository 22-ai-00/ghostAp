"""Tests for src/agent_session.py.

Covers:
- SyncClaudeCLISession: lifecycle, send_prompt, cancel, snapshot
- classify_model_failure: compaction/loop/failover detection
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Import acp.models first to break circular import chain
from src.acp.models import ACPEvent, ACPEventType, PromptResult
from src.agent_session.claude_cli import SyncClaudeCLISession
from src.agent_session.factory import create_auxiliary_session, create_engine_session
from src.agent_session.model_diagnostics import classify_model_failure
from src.agent_session.wrappers import ModelFailureAwareSession


@pytest.mark.parametrize(
    "agent_type",
    ["coco", "claude", "aiden", "codex", "gemini", "traex", "grok"],
)
def test_engine_factory_supports_every_programming_backend(
    agent_type: str,
    tmp_path,
) -> None:
    base = MagicMock()
    settings = MagicMock(rate_limit_retry_enabled=False)
    with (
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch(
            "src.agent_session.factory.normalize_acp_model_name",
            side_effect=lambda _agent, model: model,
        ),
        patch(
            "src.agent_session.factory._start_base_session",
            return_value=base,
        ) as start,
    ):
        session = create_engine_session(
            agent_type,
            str(tmp_path),
            model_name="selected-model",
        )

    assert isinstance(session, ModelFailureAwareSession)
    assert session._inner is base
    args, kwargs = start.call_args
    assert args == (agent_type, str(tmp_path), "selected-model")
    assert kwargs["allow_cli"] is True


def test_auxiliary_factory_rejects_claude_before_acp_resolution(tmp_path) -> None:
    resolver = MagicMock(
        side_effect=AssertionError("auxiliary Claude must fail before input resolution")
    )
    provider_resolution = MagicMock(
        side_effect=AssertionError("auxiliary Claude must not resolve ACP providers")
    )
    auto_update = MagicMock(
        side_effect=AssertionError("auxiliary Claude must not trigger auto-update")
    )
    with (
        patch("src.agent_session.factory._resolve_inputs", resolver),
        patch("src.acp.providers.get_providers", provider_resolution),
        patch("src.acp.sync_adapter._resolve_with_auto_update", auto_update),
    ):
        with pytest.raises(
            RuntimeError,
            match="Claude CLI backend does not support auxiliary ACP transport",
        ):
            create_auxiliary_session("claude", str(tmp_path))

    resolver.assert_not_called()
    provider_resolution.assert_not_called()
    auto_update.assert_not_called()


# ── SyncClaudeCLISession ─────────────────────────────────────────────


class TestSyncClaudeCLISession:


    def test_start_raises_when_no_executable(self):
        with patch("shutil.which", return_value=None):
            sess = SyncClaudeCLISession(cwd="/tmp")
            with pytest.raises(RuntimeError, match="未找到 Claude CLI"):
                sess.start()


    def test_cancel_sets_event(self):
        sess = SyncClaudeCLISession(cwd="/tmp")
        assert not sess._cancel_event.is_set()
        sess.cancel()
        assert sess._cancel_event.is_set()

    def test_explicit_user_cancel_is_bound_to_active_prompt_generation(
        self,
        monkeypatch,
    ):
        sess = SyncClaudeCLISession(cwd="/tmp")
        sess.session_id = "test-session"

        def cancelled_turn(*_args, **_kwargs):
            generation = sess.active_prompt_generation()
            assert generation is not None
            sess.mark_user_cancel(generation)
            return PromptResult(stop_reason="cancelled")

        monkeypatch.setattr(sess, "_send_prompt_once", cancelled_turn, raising=False)

        result = sess.send_prompt("stop this turn")

        assert result.cancellation_source == "user"
        assert sess.active_prompt_generation() is None


    def test_send_prompt_auto_starts(self):
        """send_prompt auto-calls start() when session_id is empty."""
        sess = SyncClaudeCLISession(cwd="/tmp")
        assert sess.session_id == ""

        mock_proc = MagicMock()
        mock_proc.stdout = iter(["hello world\n"])
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("src.utils.env.build_clean_env", return_value={}),
        ):
            result = sess.send_prompt("test")
            assert sess.session_id != ""
            assert result.stop_reason == "end_turn"
            assert "hello world" in result.text

    def test_send_prompt_collects_events(self):
        """on_event callback receives TEXT_CHUNK events."""
        sess = SyncClaudeCLISession(cwd="/tmp")
        sess.session_id = "test-id"

        mock_proc = MagicMock()
        mock_proc.stdout = iter(["line1\n", "line2\n"])
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        events: list[ACPEvent] = []

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("src.utils.env.build_clean_env", return_value={}),
        ):
            sess.send_prompt("test", on_event=events.append)

        assert len(events) == 2
        assert all(e.event_type == ACPEventType.TEXT_CHUNK for e in events)
        assert sess.message_count == 1

    def test_send_prompt_accepts_workflow_activity_timeout_contract(self):
        """Claude CLI must implement the common Workflow prompt signature."""
        sess = SyncClaudeCLISession(cwd="/tmp")
        sess.session_id = "test-id"

        mock_proc = MagicMock()
        mock_proc.stdout = iter(["working\n", "done\n"])
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_proc.wait = MagicMock()
        observed: list[str] = []

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("src.utils.env.build_clean_env", return_value={}),
        ):
            result = sess.send_prompt(
                "test",
                on_event=lambda event: observed.append(event.text),
                timeout=30,
                idle_timeout=5.0,
                activity_predicate=lambda event: bool(event.text.strip()),
            )

        assert result.stop_reason == "end_turn"
        assert observed == ["working\n", "done\n"]

    def test_send_prompt_emits_new_local_image_before_return(self, tmp_path):
        sess = SyncClaudeCLISession(cwd=str(tmp_path))
        sess.session_id = "test-id"
        generated = tmp_path / "screenshots" / "claude.png"

        class CreatingStdout:
            def __iter__(self):
                generated.parent.mkdir()
                generated.write_bytes(b"\x89PNG\r\n\x1a\nclaude")
                yield "created `screenshots/claude.png`\n"

        mock_proc = MagicMock()
        mock_proc.stdout = CreatingStdout()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        events: list[ACPEvent] = []

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("src.utils.env.build_clean_env", return_value={}),
        ):
            result = sess.send_prompt("take screenshot", on_event=events.append)

        assert result.stop_reason == "end_turn"
        assert events[-1].event_type == ACPEventType.IMAGE_CHUNK
        assert events[-1].image is not None
        assert events[-1].image.source_uri == str(generated)

    def test_send_prompt_does_not_emit_unreferenced_new_local_image(
        self,
        tmp_path,
    ):
        sess = SyncClaudeCLISession(cwd=str(tmp_path))
        sess.session_id = "test-id"
        generated = tmp_path / "private.png"

        class CreatingStdout:
            def __iter__(self):
                generated.write_bytes(b"\x89PNG\r\n\x1a\nprivate")
                yield "done\n"

        mock_proc = MagicMock()
        mock_proc.stdout = CreatingStdout()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        events: list[ACPEvent] = []

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("src.utils.env.build_clean_env", return_value={}),
        ):
            sess.send_prompt("work", on_event=events.append)

        assert [event.event_type for event in events] == [
            ACPEventType.TEXT_CHUNK,
        ]


# ── SyncClaudeCLISession: argument injection guard ───────────────────


class TestClaudeCLIArgInjectionGuard:
    """A5 security hardening: user text must never be parsed as CLI flags."""

    def _get_build_args(self, text: str) -> list[str]:
        """Helper to invoke the inner _build_args closure via send_prompt scaffolding."""
        from src.agent_session.claude_cli import ClaudeCLIConfig

        sess = SyncClaudeCLISession(
            cwd="/tmp",
            config=ClaudeCLIConfig(bypass_permissions=False),
        )
        sess.session_id = "test-session"

        # Access _build_args indirectly: replicate its logic since it's a closure.
        # Instead, we patch Popen to capture the args it receives.
        captured_args: list[list[str]] = []

        def fake_popen(args, **kwargs):
            captured_args.append(args)
            mock_proc = MagicMock()
            mock_proc.stdout = iter([])
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.wait = MagicMock()
            return mock_proc

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("src.utils.env.build_clean_env", return_value={}),
        ):
            sess.send_prompt(text)

        assert len(captured_args) == 1
        return captured_args[0]

    def test_double_dash_precedes_user_text(self):
        """The POSIX '--' separator must appear immediately before user text."""
        args = self._get_build_args("hello world")
        # Find user text position
        text_idx = args.index("hello world")
        assert text_idx > 0
        assert args[text_idx - 1] == "--"

    def test_dash_prefixed_text_not_treated_as_flag(self):
        """Text starting with '--help' must still be placed after '--'."""
        args = self._get_build_args("--help")
        text_idx = args.index("--help")
        # The '--' separator must be right before it
        assert args[text_idx - 1] == "--"
        # '--help' should only appear once (as user text, not as a flag)
        assert args.count("--help") == 1

    def test_dash_v_text_not_treated_as_flag(self):
        """Text '-v' must be placed after '--' separator."""
        args = self._get_build_args("-v")
        text_idx = args.index("-v")
        assert args[text_idx - 1] == "--"


# ── classify_model_failure ───────────────────────────────────────────


class TestClassifyModelFailure:
    def test_need_compaction(self):
        err = RuntimeError("Model failed: model 'gpt-5.2': receive message: need compaction")
        result = classify_model_failure(error=err)
        assert result["fail_phase"] == "model_compaction"
        assert result["reason"] == "need_compaction"
        assert result["failed_model"] == "gpt-5.2"

    def test_loop_detected(self):
        err = RuntimeError("loop detected in conversation")
        result = classify_model_failure(error=err)
        assert result["fail_phase"] == "model_loop"
        assert result["reason"] == "loop_detected"

    def test_failover(self):
        err = RuntimeError("Model failed: model 'gpt-5.2'. Failing over to: gpt-5.1")
        result = classify_model_failure(error=err)
        assert result["failed_model"] == "gpt-5.2"
        assert result["failover_to"] == "gpt-5.1"

    def test_unknown_error(self):
        err = RuntimeError("something else entirely")
        result = classify_model_failure(error=err)
        assert result["fail_phase"] == "unknown"
        assert result["reason"] == "unknown"
