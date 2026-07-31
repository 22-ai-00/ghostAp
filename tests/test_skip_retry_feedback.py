"""Focused skip-retry UI text and retry callback contract tests."""

from unittest.mock import MagicMock

from src.card.ui_text import UI_TEXT


class TestSkipRetryUITextKeys:
    """Guard: skip_retry_ack and no_active_retry keys exist in UI_TEXT."""

    def test_skip_retry_feedback_texts_are_present(self):
        for key in ("skip_retry_ack", "no_active_retry"):
            assert UI_TEXT.get(key, "").strip(), f"{key} must be a non-empty UI text"


# ---------------------------------------------------------------------------
# Task 21: Retry success callback sequence
# ---------------------------------------------------------------------------


class TestRetrySuccessCallbackSequence:
    """on_retry_status fires WAITING→EXECUTING→SUCCEEDED in correct order."""

    def test_callback_sequence_on_success(self):
        """Successful retry emits WAITING, EXECUTING, SUCCEEDED in order."""
        from src.spec_engine.retry_status import RetryEvent, RetryStatus
        from src.spec_engine.review_retry import PipelineRetryContext, attempt_pipeline_retry
        from src.spec_engine.review_types import ReviewCircuitState

        circuit = MagicMock(spec=ReviewCircuitState)
        circuit.consecutive_timeouts = 1

        settings = MagicMock()
        settings.spec_review_retry_max_attempts = 1
        settings.spec_review_retry_max_delay = 3  # >= 2 to trigger WAITING event
        settings.spec_review_retry_base_delay = 3.0
        settings.spec_review_retry_decay_factor = 1.5
        settings.spec_review_min_timeout = 30
        settings.spec_review_hard_floor = 15

        mock_outcome = MagicMock()
        mock_outcome.error = None
        mock_outcome.error_code = None
        mock_outcome.review = MagicMock()
        pipeline_fn = MagicMock(return_value=[mock_outcome])

        events_received: list = []

        def on_status(event: RetryEvent):
            events_received.append(event.status)

        ctx = PipelineRetryContext(
            cancel_event=None,
            on_retry_status=on_status,
            base_timeout=120,
            multiplier=2,
            pipeline_fn=pipeline_fn,
            budget_cls=MagicMock(),
            artifacts=MagicMock(),
            agent_type="test",
            model_name=None,
            skip_retry_event=None,
        )

        from unittest.mock import patch as _patch
        with _patch("src.spec_engine.review_retry.time.sleep", return_value=None):
            result = attempt_pipeline_retry(circuit=circuit, settings=settings, cycle=1, ctx=ctx)

        assert result is not None
        # Verify sequence: WAITING (because delay >= 2), EXECUTING, SUCCEEDED
        assert RetryStatus.WAITING in events_received
        assert RetryStatus.EXECUTING in events_received
        assert RetryStatus.SUCCEEDED in events_received
        # Order: WAITING before EXECUTING before SUCCEEDED
        w_idx = events_received.index(RetryStatus.WAITING)
        e_idx = events_received.index(RetryStatus.EXECUTING)
        s_idx = events_received.index(RetryStatus.SUCCEEDED)
        assert w_idx < e_idx < s_idx
