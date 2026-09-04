"""Tests for AC4 isolation: intermediate-result boundaries in Workflow.

AC4 is the contract that prevents workflow intermediate results from leaking
into the main agent chat context.  It has three enforcement surfaces:

1. ``on_agent_done`` callback payloads contain only meta-information
   (label/tool/model/token_usage/duration_s/cached/error) and never include
   ``output`` or ``parsed`` fields — so callers cannot accidentally inject
   intermediate outputs into the main chat.
2. ``WorkflowStateManager.add_context_tokens`` is the *only* authorised
   counter of how many characters the workflow pushes into the main chat.
   The engine calls it only from the final-result path.
3. ``WorkflowRenderer`` rejects explicitly configured forbidden sentinels in
   user-visible card projections. Safe, redacted execution blocks are allowed
   in Workflow cards; the isolation boundary is the main-agent context and
   callback payload, not blanket suppression from the initiating user's UI.

Together these three surface-tests pin down the isolation invariant:
``delta_context_tokens ≈ len(final_result)`` and nothing else.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock

from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    WorkflowMetrics,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.state_manager import WorkflowStateManager

# ===========================================================================
# 1. on_agent_done payload stripped of output/parsed
# ===========================================================================


class TestAC4AgentDonePayloadStripped(unittest.TestCase):
    """Verify ``on_agent_done`` callbacks carry meta-only payloads."""

    def _make_engine(self, on_agent_done_cb=None):
        """Build a lightweight WorkflowEngine wired for callback observation."""
        from src.workflow_engine.engine import WorkflowEngine
        from src.workflow_engine.renderer import WorkflowProgressRenderer
        from src.workflow_engine.state_manager import WorkflowStateManager

        engine = WorkflowEngine.__new__(WorkflowEngine)
        engine._lock = threading.Lock()
        engine._project = WorkflowProject(
            workflow_id="wf-ac4-test",
            status=WorkflowStatus.RUNNING,
            metrics=WorkflowMetrics(),
        )
        engine._state_manager = WorkflowStateManager(engine._project)
        engine._cancel_event = threading.Event()
        engine._agent_call_count = 0
        engine._journal = None
        engine._progress_coalescer = None
        engine._renderer_wf = MagicMock(spec=WorkflowProgressRenderer)

        # Callbacks — the thing under test.
        from src.workflow_engine.engine import WorkflowEngineCallbacks

        callbacks = WorkflowEngineCallbacks()
        callbacks.on_agent_done = on_agent_done_cb
        engine._callbacks = callbacks

        # Executor returns a fake "sensitive" result so we can prove the
        # payload truly drops output/parsed rather than never having them.
        engine._executor = MagicMock()
        engine._executor.execute.return_value = AgentCallResult(
            output="SENSITIVE_INTERMEDIATE_OUTPUT_XYZ",
            parsed={"sensitive_key": "sensitive_value"},
            token_usage=128,
            duration_s=0.1,
            tool="coco",
            model="claude",
        )
        return engine

    def test_on_agent_done_payload_has_no_output_or_parsed(self):
        """Manual engine callback must not leak output/parsed into payload."""
        observed: list[tuple[str, dict]] = []

        def cb(label: str, payload: dict) -> None:
            observed.append((label, payload))

        engine = self._make_engine(on_agent_done_cb=cb)

        params = AgentCallParams(
            prompt="does not matter", tool="coco", model="claude", label="ac4-unit"
        )
        result = engine._handle_agent_call(params)

        # Sanity: a real result was produced, confirming output/parsed were
        # available to the engine but intentionally dropped.
        self.assertEqual(result.output, "SENSITIVE_INTERMEDIATE_OUTPUT_XYZ")
        self.assertEqual(result.parsed, {"sensitive_key": "sensitive_value"})

        self.assertEqual(len(observed), 1)
        label, payload = observed[0]
        self.assertEqual(label, "ac4-unit")

        # The forbidden keys must not appear — even if set to None.
        self.assertNotIn("output", payload)
        self.assertNotIn("parsed", payload)

        # The expected meta-info keys are present.
        self.assertEqual(payload.get("label"), "ac4-unit")
        self.assertEqual(payload.get("tool"), "coco")
        self.assertEqual(payload.get("token_usage"), 128)
        self.assertAlmostEqual(payload.get("duration_s"), 0.1, places=5)
        self.assertFalse(payload.get("cached"))
        self.assertIsNone(payload.get("error"))

    def test_cache_hit_payload_has_no_output_or_parsed(self):
        """On cache-hit path the state manager's on_agent_done records only
        meta info (token_usage/duration_s/cached), never output/parsed."""
        observed_sm: list[tuple[str, dict]] = []
        observed_callback: list[tuple[str, dict]] = []

        engine = self._make_engine(
            on_agent_done_cb=lambda label, payload: observed_callback.append(
                (label, payload)
            )
        )
        original_sm_done = engine._state_manager.on_agent_done

        def spy_on_agent_done(label: str, result: dict) -> None:
            observed_sm.append((label, result))
            original_sm_done(label, result)

        engine._state_manager.on_agent_done = spy_on_agent_done

        # Drive the cache path: attach a journal, store a result under the
        # key the engine will compute from params, then call _handle_agent_call.
        import tempfile

        from src.workflow_engine.journal import WorkflowJournal

        engine._journal = WorkflowJournal(root_path=tempfile.mkdtemp(), run_id="ac4")
        params = AgentCallParams(
            prompt="same prompt", tool="coco", model="claude", label="ac4-cache"
        )
        key = engine._journal.compute_key(params.prompt, params.tool, params.model)
        engine._journal.store(
            key,
            AgentCallResult(
                output="SENSITIVE_CACHED_OUTPUT",
                parsed={"cached_key": "cached_value"},
                token_usage=500,
                duration_s=2.0,
                tool="coco",
                model="claude",
            ),
        )

        result = engine._handle_agent_call(params)

        # Sanity: cache path returned the cached output.
        self.assertTrue(result.cached)
        self.assertEqual(result.output, "SENSITIVE_CACHED_OUTPUT")

        # Verify state manager was notified with meta-only payload.
        self.assertGreaterEqual(len(observed_sm), 1)
        label, sm_payload = observed_sm[-1]
        self.assertEqual(label, "ac4-cache")
        # No output/parsed keys.
        self.assertNotIn("output", sm_payload)
        self.assertNotIn("parsed", sm_payload)
        # Must include token_usage/duration/cached flags.
        self.assertIn("token_usage", sm_payload)
        self.assertIn("duration_s", sm_payload)
        self.assertTrue(sm_payload.get("cached"))
        self.assertEqual(len(observed_callback), 1)
        callback_label, callback_payload = observed_callback[0]
        self.assertEqual(callback_label, "ac4-cache")
        self.assertTrue(callback_payload.get("cached"))
        self.assertNotIn("output", callback_payload)
        self.assertNotIn("parsed", callback_payload)


# ===========================================================================
# 2. delta_context_tokens: only final-result path grows it
# ===========================================================================


class TestAC4DeltaContextTokensOnlyFinalResult(unittest.TestCase):
    """Test that ``delta_context_tokens`` is only incremented via
    ``add_context_tokens`` and stays at zero on the ``on_agent_done`` paths.
    """

    def test_add_context_tokens_accumulates(self):
        project = WorkflowProject()
        mgr = WorkflowStateManager(project)
        mgr.add_context_tokens(500)
        self.assertEqual(mgr.delta_context_tokens, 500)
        mgr.add_context_tokens(300)
        self.assertEqual(mgr.delta_context_tokens, 800)

    def test_on_agent_done_does_not_grow_delta_context_tokens(self):
        """Intermediate-agent completion must leave the audit counter alone."""
        project = WorkflowProject()
        mgr = WorkflowStateManager(project)
        mgr.on_phase_changed("P1")

        # Start + finish a handful of intermediate agents.  None may inflate
        # delta_context_tokens — only the final-result path does.
        for i in range(4):
            label = f"agent-{i}"
            mgr.on_agent_started(label, "coco", "P1")
            mgr.on_agent_done(
                label, {"token_usage": 100 * (i + 1), "duration_s": 0.1 * (i + 1)}
            )

        self.assertEqual(mgr.delta_context_tokens, 0)
        # Token totals are tracked in metrics (must still be correct).
        self.assertEqual(project.metrics.total_tokens, 100 + 200 + 300 + 400)

    def test_add_context_tokens_clamps_negative_to_zero(self):
        """Negative inputs must be clamped — only non-negative accumulation."""
        project = WorkflowProject()
        mgr = WorkflowStateManager(project)
        mgr.add_context_tokens(-500)
        self.assertEqual(mgr.delta_context_tokens, 0)
        mgr.add_context_tokens(100)
        mgr.add_context_tokens(-9999)
        self.assertEqual(mgr.delta_context_tokens, 100)

    def test_concurrent_add_context_tokens_is_atomic(self):
        project = WorkflowProject()
        mgr = WorkflowStateManager(project)
        n_threads = 8
        per_thread = 200

        def worker():
            for _ in range(per_thread):
                mgr.add_context_tokens(1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(mgr.delta_context_tokens, n_threads * per_thread)


# ===========================================================================
# 3. Renderer card-leak sentinel
# ===========================================================================


class TestAC4ProgressCardNoLeakage(unittest.TestCase):
    """Renderer must not expose raw intermediate-agent error text."""

    def test_progress_card_omits_agent_error_sentinel(self):
        from src.workflow_engine.models import AgentStatus
        from src.workflow_engine.renderer import WorkflowProgressRenderer

        sentinel = "AC4_SENTINEL_RESULT_must_not_leak"
        project = WorkflowProject(
            status=WorkflowStatus.RUNNING,
            metrics=WorkflowMetrics(),
        )
        mgr = WorkflowStateManager(project)
        mgr.on_phase_changed("Phase 1")
        mgr.on_agent_started("a1", "coco", "Phase 1")
        # Set the agent to FAILED and inject a sentinel string into its error
        # field. Progress cards expose only the generic terminal operation;
        # raw errors belong in the final result ledger.
        project.phases[0].agents[0].status = AgentStatus.FAILED
        project.phases[0].agents[0].error = sentinel

        card = WorkflowProgressRenderer(project).render_progress_card()

        self.assertNotIn(sentinel, json.dumps(card, ensure_ascii=False))

    def test_completion_card_raises_on_sentinel(self):
        import src.workflow_engine.renderer as renderer_module
        from src.workflow_engine.renderer import render_completion_card

        project = WorkflowProject(
            status=WorkflowStatus.FAILED,
            result="OK",
            metrics=WorkflowMetrics(),
        )
        # project.error is only rendered for FAILED workflows.
        project.error = "AC4_SENTINEL_RESULT_completion_message"

        original = renderer_module._AGENT_OUTPUT_FORBIDDEN_MARKERS
        try:
            renderer_module._AGENT_OUTPUT_FORBIDDEN_MARKERS = (
                "AC4_SENTINEL_RESULT_completion_message",
            )
            with self.assertRaises(RuntimeError, msg="card leaked agent output"):
                render_completion_card(project)
        finally:
            renderer_module._AGENT_OUTPUT_FORBIDDEN_MARKERS = original

    def test_progress_card_without_sentinel_is_fine(self):
        """Cards without forbidden markers render without raising."""
        import src.workflow_engine.renderer as renderer_module
        from src.workflow_engine.renderer import WorkflowProgressRenderer

        project = WorkflowProject(
            status=WorkflowStatus.RUNNING,
            metrics=WorkflowMetrics(),
        )
        mgr = WorkflowStateManager(project)
        mgr.on_phase_changed("Phase 1")
        mgr.on_agent_started("a1", "coco", "Phase 1")
        mgr.on_agent_done("a1", {"token_usage": 10, "duration_s": 0.1})

        original = renderer_module._AGENT_OUTPUT_FORBIDDEN_MARKERS
        try:
            renderer_module._AGENT_OUTPUT_FORBIDDEN_MARKERS = (
                "AC4_SENTINEL_RESULT_NOPE",
            )
            r = WorkflowProgressRenderer(project)
            card = r.render_progress_card()
            self.assertIsInstance(card, dict)
        finally:
            renderer_module._AGENT_OUTPUT_FORBIDDEN_MARKERS = original


# ===========================================================================
# 4. Ratio: delta_context_tokens ≈ len(final_result)
# ===========================================================================


class TestAC4Ratio(unittest.TestCase):
    """End-to-end shape: ``delta_context_tokens`` should track final-result
    length closely; spurious intermediate contributions show up as a bad
    ratio and demonstrate an AC4-regression scenario."""

    def test_delta_matches_final_result_length(self):
        result_text = "Hello world — the final, compact workflow output."
        project = WorkflowProject()
        mgr = WorkflowStateManager(project)

        # Simulate the only legal call site (the engine's final-result path).
        mgr.add_context_tokens(len(result_text or ""))

        # Allow generous 1.5× slack — the contract is "on the order of
        # len(result)", not "exactly equal".  Any intermediate call to
        # add_context_tokens would immediately blow through this bound.
        self.assertLessEqual(mgr.delta_context_tokens, int(1.5 * len(result_text)))
        self.assertGreater(mgr.delta_context_tokens, 0)

    def test_intermediate_add_context_tokens_regression_scenario(self):
        """Document the bad case: a malicious/mistaken intermediate agent that
        called ``add_context_tokens(10_000)`` would blow the ratio — this test
        records the failure mode rather than asserting it must pass.  It's a
        negative-spec guard: if someone changes add_context_tokens to silently
        ignore large values, this test will detect it."""
        project = WorkflowProject()
        mgr = WorkflowStateManager(project)

        # Good final-result contribution.
        final_result_text = "Hello world"
        mgr.add_context_tokens(len(final_result_text))

        # Bad: an intermediate agent's output is wrongly fed into the counter.
        mgr.add_context_tokens(10_000)

        # The ratio is now absurd — document the violation.
        ratio = mgr.delta_context_tokens / max(1, len(final_result_text))
        # The point: ratio is huge, not ≈ 1.  We assert the obvious lower
        # bound to pin down the bad case — the intent is that *no* such
        # intermediate call should exist in production code paths.
        self.assertGreaterEqual(ratio, 100,
            "Bad scenario: intermediate add_context_tokens inflates ratio. "
            "Production paths must never call it from within agent callbacks.")

        # And we confirm the property that add_context_tokens *does* grow the
        # counter: once a bad call exists, the audit surface catches it.
        self.assertGreater(mgr.delta_context_tokens, int(1.5 * len(final_result_text)))


if __name__ == "__main__":
    unittest.main()
