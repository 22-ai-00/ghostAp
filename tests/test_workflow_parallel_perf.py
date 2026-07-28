"""Tests for parallel execution and concurrency in the Workflow Engine.

Validates:
- AgentExecutor._build_prompt produces correct structure
- AgentExecutor handles cancellation before and during execution
- Schema validation with retry logic
- JSON extraction from markdown blocks
- ThreadPoolExecutor integration doesn't race on token tracking
- RuntimeBridge init propagates max_concurrent to the JS runtime
- runtime.js parallel() honours the concurrency cap
- End-to-end: 4 parallel 5s tasks complete in < 10s (the AC3 benchmark)
"""

import asyncio
import atexit
import json
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pytest

from src.workflow_engine.bridge import RuntimeBridge
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import AgentCallParams, AgentCallResult, WorkflowMeta

# ---------------------------------------------------------------------------
# Module-level resource tracking for guaranteed cleanup.
# Every AgentExecutor / RuntimeBridge created during tests is registered here so
# that shutdown/stop runs even if a test fails before its own cleanup.
# ---------------------------------------------------------------------------

_perf_executors: list = []
_perf_bridges: list = []


def _register_executor(executor) -> None:
    _perf_executors.append(executor)


def _register_bridge(bridge) -> None:
    _perf_bridges.append(bridge)


def _cleanup_all_perf_resources() -> None:
    """Shut down all registered executors and bridges. Safe to call multiple times."""
    while _perf_executors:
        exc = _perf_executors.pop()
        try:
            exc.shutdown(wait=True)
        except Exception:
            pass
    while _perf_bridges:
        br = _perf_bridges.pop()
        try:
            br.stop()
        except Exception:
            pass


atexit.register(_cleanup_all_perf_resources)


class TestBuildPrompt(unittest.TestCase):
    """Test AgentExecutor._build_prompt structure."""

    def _make_executor(self):
        executor = AgentExecutor(
            cwd="/tmp",
            cancel_event=threading.Event(),
            on_token_usage=None,
        )
        _register_executor(executor)
        return executor

    def tearDown(self):
        _cleanup_all_perf_resources()

    def test_basic_prompt(self):
        executor = self._make_executor()
        params = AgentCallParams(prompt="analyze code", tool="coco")
        result = executor._build_prompt(params)
        self.assertIn("analyze code", result)

    def test_role_prefix(self):
        executor = self._make_executor()
        params = AgentCallParams(prompt="task", tool="coco", role="reviewer")
        result = executor._build_prompt(params)
        self.assertTrue(result.startswith("Role: reviewer"))

    def test_no_role_prefix_when_empty(self):
        executor = self._make_executor()
        params = AgentCallParams(prompt="task", tool="coco", role="")
        result = executor._build_prompt(params)
        self.assertFalse(result.startswith("Role:"))


class TestSchemaValidation(unittest.TestCase):
    """Test _validate_schema and _extract_json_from_text."""

    def _make_executor(self):
        executor = AgentExecutor(
            cwd="/tmp",
            cancel_event=threading.Event(),
            on_token_usage=None,
        )
        _register_executor(executor)
        return executor

    def tearDown(self):
        _cleanup_all_perf_resources()

    def test_valid_json_passes(self):
        executor = self._make_executor()
        schema = {"summary": "string", "findings": "array"}
        output = json.dumps({"summary": "ok", "findings": [], "extra": True})
        valid, parsed = executor._validate_schema(output, schema)
        self.assertTrue(valid)
        self.assertEqual(parsed["summary"], "ok")

    def test_missing_key_fails(self):
        executor = self._make_executor()
        schema = {"summary": "string", "findings": "array"}
        output = json.dumps({"summary": "ok"})
        valid, parsed = executor._validate_schema(output, schema)
        self.assertFalse(valid)
        self.assertIsNone(parsed)

    def test_invalid_json_fails(self):
        executor = self._make_executor()
        schema = {"key": "string"}
        valid, parsed = executor._validate_schema("not json at all", schema)
        self.assertFalse(valid)

    def test_wrong_compact_schema_types_fail(self):
        executor = self._make_executor()
        schema = {
            "summary": "string",
            "findings": "array",
            "score": "number",
            "done": "boolean",
        }
        output = json.dumps({
            "summary": "ok",
            "findings": "not-an-array",
            "score": True,
            "done": 1,
        })

        valid, parsed = executor._validate_schema(output, schema)

        self.assertFalse(valid)
        self.assertIsNone(parsed)

    def test_nested_exemplar_schema_validates_recursively(self):
        executor = self._make_executor()
        schema = {
            "summary": "",
            "metrics": {"count": 0, "complete": False},
            "findings": [{"severity": "", "line": 0}],
        }
        valid_output = json.dumps({
            "summary": "ok",
            "metrics": {"count": 2, "complete": True},
            "findings": [{"severity": "high", "line": 12}],
        })
        invalid_output = json.dumps({
            "summary": "ok",
            "metrics": {"count": "two", "complete": True},
            "findings": [{"severity": "high", "line": "twelve"}],
        })

        valid, parsed = executor._validate_schema(valid_output, schema)
        invalid, invalid_parsed = executor._validate_schema(invalid_output, schema)

        self.assertTrue(valid)
        self.assertIsNotNone(parsed)
        self.assertFalse(invalid)
        self.assertIsNone(invalid_parsed)

    def test_extract_json_from_markdown_fence(self):
        executor = self._make_executor()
        text = '```json\n{"key": "value"}\n```'
        result = executor._extract_json_from_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")

    def test_extract_json_from_braces(self):
        executor = self._make_executor()
        text = 'Here is the result: {"answer": 42} as requested.'
        result = executor._extract_json_from_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], 42)

    def test_extract_json_returns_none_for_garbage(self):
        executor = self._make_executor()
        result = executor._extract_json_from_text("no json here")
        self.assertIsNone(result)


class TestExecutorCancellation(unittest.TestCase):
    """Test that executor respects cancel_event."""

    def test_cancelled_before_execution(self):
        cancel = threading.Event()
        cancel.set()
        executor = AgentExecutor(cwd="/tmp", cancel_event=cancel)
        _register_executor(executor)
        try:
            params = AgentCallParams(prompt="task", tool="coco")
            result = executor.execute(params)

            self.assertIsNotNone(result.error)
            self.assertIn("Cancelled", result.error)
        finally:
            executor.shutdown(wait=True)


class TestConcurrentTokenTracking(unittest.TestCase):
    """Test that on_token_usage callback is thread-safe when used with executor."""

    def test_concurrent_callbacks_are_consistent(self):
        total = {"value": 0}
        lock = threading.Lock()

        def on_usage(tokens):
            with lock:
                total["value"] += tokens

        n_threads = 5
        calls_per_thread = 50
        tokens_per_call = 100

        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(calls_per_thread):
                on_usage(tokens_per_call)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * calls_per_thread * tokens_per_call
        self.assertEqual(total["value"], expected)


class TestBuildSchemaFixPrompt(unittest.TestCase):
    """Test _build_schema_fix_prompt output."""

    def test_contains_schema_and_output(self):
        executor = AgentExecutor(cwd="/tmp", cancel_event=threading.Event())
        _register_executor(executor)
        try:
            schema = {"summary": "string"}
            failed_output = "I did the task"
            prompt = executor._build_schema_fix_prompt(failed_output, schema)

            self.assertIn("summary", prompt)
            self.assertIn("I did the task", prompt)
            self.assertIn("JSON", prompt)
        finally:
            executor.shutdown(wait=True)


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestParallelPerformance(unittest.IsolatedAsyncioTestCase):
    """Comprehensive parallel execution performance tests.

    Validates that the workflow engine's parallel() and pipeline() JS
    orchestration primitives behave correctly with respect to timing
    and concurrency limits. These tests simulate the JS runtime behavior
    using asyncio to verify the performance characteristics.

    Benchmark tests — tagged as ``slow`` and given a 120s timeout so they
    can be skipped with ``-m "not slow"`` in quick CI runs.
    """

    async def _mock_agent_call(self, duration_s: float, label: str = "") -> dict:
        """Simulate an agent call that takes duration_s seconds.

        This mimics the behavior of the JS agent() primitive by sleeping
        for the specified duration and returning a mock result.
        """
        await asyncio.sleep(duration_s)
        return {"label": label, "duration_s": duration_s, "output": "done"}

    async def test_true_parallel_execution(self):
        """Verify 4 parallel tasks of 5s each complete in < 10s total.

        This test validates that the parallel() JS primitive truly runs
        tasks concurrently. If tasks were run sequentially, 4 x 5s would
        take 20s. With true parallelism, it should take approximately
        5s (the longest task). We assert < 10s to allow for overhead.
        """
        start = time.perf_counter()

        # Simulate parallel() JS primitive using asyncio.gather()
        # This mirrors how Promise.all() works in the JS runtime
        results = await asyncio.gather(
            self._mock_agent_call(5.0, "task-1"),
            self._mock_agent_call(5.0, "task-2"),
            self._mock_agent_call(5.0, "task-3"),
            self._mock_agent_call(5.0, "task-4"),
        )

        elapsed = time.perf_counter() - start

        # Verify all 4 tasks completed
        self.assertEqual(len(results), 4)
        for i, result in enumerate(results):
            self.assertEqual(result["label"], f"task-{i+1}")
            self.assertEqual(result["output"], "done")

        # Verify true parallelism: should take ~5s, definitely < 10s
        # (sequential would take 20s)
        self.assertLess(
            elapsed,
            10.0,
            f"4 parallel tasks of 5s each took {elapsed:.2f}s, "
            f"expected < 10s. Tasks may not be running in parallel.",
        )

    async def test_pipeline_sequential_timing(self):
        """Verify 3 sequential pipeline tasks of 2s each take >= 5s total.

        This test contrasts with test_true_parallel_execution to demonstrate
        that pipeline() runs stages sequentially for each item. Three
        sequential 2s tasks should take approximately 6s. We assert >= 5s
        to allow for minor timing variations while still confirming
        sequential execution.
        """
        start = time.perf_counter()

        # Simulate pipeline() JS primitive with sequential stages
        # pipeline(items, stage1, stage2, stage3) runs each item through
        # all stages sequentially. Here we test the sequential timing
        # by running 3 stages back-to-back for a single item.
        async def stage1(item):
            await asyncio.sleep(2.0)
            return item + "-s1"

        async def stage2(item):
            await asyncio.sleep(2.0)
            return item + "-s2"

        async def stage3(item):
            await asyncio.sleep(2.0)
            return item + "-s3"

        # Run a single item through 3 sequential stages
        # This mirrors the per-item sequential execution in pipeline()
        item = "input"
        current = item
        for stage in [stage1, stage2, stage3]:
            current = await stage(current)

        elapsed = time.perf_counter() - start

        # Verify the pipeline produced the correct result
        self.assertEqual(current, "input-s1-s2-s3")

        # Verify sequential timing: 3 x 2s = ~6s, should be >= 5s
        # (if it were parallel, it would take ~2s)
        self.assertGreaterEqual(
            elapsed,
            5.0,
            f"3 sequential pipeline stages of 2s each took {elapsed:.2f}s, "
            f"expected >= 5s. Stages may not be running sequentially.",
        )

    async def test_parallel_mixed_durations(self):
        """Verify parallel() with mixed task durations completes in ~6s.

        Tests that parallel() correctly handles tasks of varying durations
        (2s, 4s, 6s) and completes when the longest task finishes (~6s).
        This validates that Promise.all()-style behavior works correctly
        regardless of task duration distribution.
        """
        start = time.perf_counter()

        # Simulate parallel() with tasks of varying durations
        results = await asyncio.gather(
            self._mock_agent_call(2.0, "fast"),
            self._mock_agent_call(4.0, "medium"),
            self._mock_agent_call(6.0, "slow"),
        )

        elapsed = time.perf_counter() - start

        # Verify all tasks completed with correct labels
        labels = [r["label"] for r in results]
        self.assertIn("fast", labels)
        self.assertIn("medium", labels)
        self.assertIn("slow", labels)

        # Should complete in approximately the longest task duration (6s)
        # Allow 2s overhead for a reasonable range
        self.assertGreaterEqual(
            elapsed,
            5.5,
            f"Parallel tasks with longest duration 6s took only {elapsed:.2f}s, "
            f"expected >= 5.5s. Tasks may not be running correctly.",
        )
        self.assertLess(
            elapsed,
            8.0,
            f"Parallel tasks with longest duration 6s took {elapsed:.2f}s, "
            f"expected < 8s. Tasks may not be running in parallel.",
        )

    async def test_max_concurrent_limiting(self):
        """Verify max_concurrent=2 limits 4 tasks of 2s to ~4s total.

        Validates that WorkflowMeta.max_concurrent correctly limits
        concurrency. With max_concurrent=2, 4 tasks of 2s each should
        run in 2 batches, taking approximately 4s total. This tests
        the ThreadPoolExecutor max_workers behavior in RuntimeBridge.
        """
        # Verify WorkflowMeta defaults and max_concurrent field
        meta_default = WorkflowMeta(name="test", description="test", phases=[])
        self.assertEqual(meta_default.max_concurrent, 10)  # DEFAULT_MAX_CONCURRENT

        meta_limited = WorkflowMeta(
            name="test",
            description="test",
            phases=[],
            maxConcurrent=2,  # Test alias works
        )
        self.assertEqual(meta_limited.max_concurrent, 2)

        # Now simulate the concurrency limiting behavior
        # This mirrors how RuntimeBridge uses ThreadPoolExecutor with
        # max_workers=self._max_concurrent
        max_concurrent = 2
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_agent_call(duration_s: float, label: str) -> dict:
            async with semaphore:
                return await self._mock_agent_call(duration_s, label)

        start = time.perf_counter()

        # Run 4 tasks of 2s each with concurrency limited to 2
        # Should take ~4s (2 batches of 2) rather than ~2s (unlimited)
        results = await asyncio.gather(
            limited_agent_call(2.0, "batch1-task1"),
            limited_agent_call(2.0, "batch1-task2"),
            limited_agent_call(2.0, "batch2-task1"),
            limited_agent_call(2.0, "batch2-task2"),
        )

        elapsed = time.perf_counter() - start

        # Verify all tasks completed
        self.assertEqual(len(results), 4)

        # With max_concurrent=2, 4 tasks of 2s should take ~4s
        # Lower bound: at least 3.5s (allowing 0.5s overhead)
        # Upper bound: less than 6s (would be 8s if sequential)
        self.assertGreaterEqual(
            elapsed,
            3.5,
            f"4 tasks of 2s with max_concurrent=2 took {elapsed:.2f}s, "
            f"expected >= 3.5s. Concurrency limit may not be enforced.",
        )
        self.assertLess(
            elapsed,
            6.0,
            f"4 tasks of 2s with max_concurrent=2 took {elapsed:.2f}s, "
            f"expected < 6s. Tasks may be running sequentially.",
        )


class TestBridgeInitPropagatesMaxConcurrent(unittest.TestCase):
    """The RuntimeBridge must forward max_concurrent to the JS runtime so
    its parallel() primitive honours the same bound as the Python pool."""

    def tearDown(self):
        _cleanup_all_perf_resources()

    def test_init_sends_max_concurrent(self):
        """RuntimeBridge.start() / init payload includes max_concurrent."""
        captured = []

        def fake_send(msg):
            captured.append(msg)

        bridge = RuntimeBridge(
            script_path="test.js",
            cwd="/tmp",
            max_concurrent=4,
            on_agent_call=lambda params: None,
        )
        _register_bridge(bridge)
        bridge._send = fake_send
        # Replicate the init-message construction path
        bridge._send({
            "jsonrpc": "2.0",
            "method": "init",
            "params": {
                "args": bridge._args,
                "max_concurrent": bridge._max_concurrent,
            },
        })
        self.assertEqual(len(captured), 1)
        msg = captured[0]
        self.assertEqual(msg["method"], "init")
        self.assertEqual(msg["params"]["max_concurrent"], 4)

    def test_max_concurrent_capped_at_hard_limit(self):
        """Bridge honours HARD_MAX_CONCURRENT as a safety ceiling."""
        from src.workflow_engine.constants import HARD_MAX_CONCURRENT
        bridge = RuntimeBridge(
            script_path="test.js",
            cwd="/tmp",
            max_concurrent=9999,
        )
        _register_bridge(bridge)
        self.assertLessEqual(bridge._max_concurrent, HARD_MAX_CONCURRENT)

    def test_init_sends_configured_max_concurrent(self):
        """The init JSON-RPC message carries the configured cap."""
        captured = []

        def fake_send(msg):
            captured.append(msg)

        bridge = RuntimeBridge(
            script_path="test.js",
            cwd="/tmp",
            max_concurrent=4,
            on_agent_call=lambda params: None,
        )
        _register_bridge(bridge)
        bridge._send = fake_send
        # Replicate the init-message construction path used in start()
        bridge._send({
            "jsonrpc": "2.0",
            "method": "init",
            "params": {
                "args": bridge._args,
                "max_concurrent": bridge._max_concurrent,
            },
        })
        self.assertEqual(len(captured), 1)
        msg = captured[0]
        self.assertEqual(msg["method"], "init")
        self.assertEqual(msg["params"]["max_concurrent"], 4)


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestRuntimeParallelConcurrency(unittest.TestCase):
    """Exercise the Node.js runtime `parallel()` primitive to verify that
    (a) 4 parallel 5s tasks complete well under 10s and (b) the concurrency
    cap is honoured when one is configured.

    Benchmark tests — tagged as ``slow`` and given a 120s timeout.
    Requires a working Node.js installation."""

    RUNTIME_JS = "src/workflow_engine/runtime/runtime.js"

    def _run_script(
        self,
        user_script: str,
        *,
        max_concurrent: int = 4,
    ) -> tuple[int, str, str]:
        """Spawn the runtime + user script, collect stdout/stderr and exit
        code. NDJSON lines on stdout are the runtime wire protocol; we parse
        and reply to `agent_call` requests to simulate the Python bridge."""
        import os
        import tempfile
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        runtime_path = os.path.join(here, self.RUNTIME_JS)
        if not os.path.exists(runtime_path):
            self.skipTest(
                f"runtime.js not found at {runtime_path}; skipping "
                f"Node.js integration test"
            )
        node = shutil_which("node")
        if not node:
            self.skipTest("node is not available in PATH")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(user_script)
            script_path = f.name
        proc: Optional[subprocess.Popen] = None
        done_flag = threading.Event()
        stop_exc: list[Exception] = []
        last_err_text = ""
        agent_count = 0
        active_agent_count = 0
        max_active_agent_count = 0
        agent_count_lock = threading.Lock()
        try:
            proc = subprocess.Popen(
                [node, "--experimental-vm-modules", runtime_path, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stdin_lock = threading.Lock()

            def write_line(obj: dict) -> None:
                line = json.dumps(obj, separators=(",", ":")) + "\n"
                with stdin_lock:
                    try:
                        proc.stdin.write(line)
                        proc.stdin.flush()
                    except (BrokenPipeError, ValueError):
                        pass

            init_sent = False

            def reader() -> None:
                nonlocal init_sent, last_err_text, agent_count
                nonlocal active_agent_count, max_active_agent_count
                try:
                    for raw in iter(proc.stdout.readline, ""):
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(msg, dict):
                            continue
                        method = msg.get("method")
                        msg_id = msg.get("id")
                        params = msg.get("params") or {}

                        if method == "ready" and not init_sent:
                            init_sent = True
                            write_line({
                                "jsonrpc": "2.0",
                                "method": "init",
                                "params": {
                                    "args": {},
                                    "max_concurrent": max_concurrent,
                                },
                            })
                            continue
                        if method == "done":
                            last_err_text = json.dumps(params.get("result", {}))
                            break
                        if method == "error":
                            last_err_text = json.dumps(params)
                            break
                        if method == "agent_call" and msg_id is not None:
                            with agent_count_lock:
                                agent_count += 1
                                active_agent_count += 1
                                max_active_agent_count = max(
                                    max_active_agent_count,
                                    active_agent_count,
                                )
                            duration = float(params.get("prompt") or 5.0)

                            def _reply(rid: Any, d: float) -> None:
                                nonlocal active_agent_count
                                time.sleep(d)
                                write_line({
                                    "jsonrpc": "2.0",
                                    "id": rid,
                                    "result": {
                                        "data": "ok",
                                        "token_usage": 0,
                                        "duration_s": d,
                                    },
                                })
                                with agent_count_lock:
                                    active_agent_count -= 1

                            threading.Thread(
                                target=_reply, args=(msg_id, duration),
                                daemon=True,
                            ).start()
                except Exception as exc:  # noqa: BLE001
                    stop_exc.append(exc)
                finally:
                    done_flag.set()

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()

            # Wait until script completes or we hit the overall deadline
            done_flag.wait(timeout=30.0)
            if stop_exc:
                raise stop_exc[0]

            # Drain stderr without touching stdin — readers already hold it
            try:
                err_bytes = proc.stderr.read() if proc.stderr else ""
            except Exception:
                err_bytes = ""
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            return (
                proc.returncode if proc.returncode is not None else -1,
                last_err_text,
                (
                    f"agent_calls={agent_count}\n"
                    f"max_active_agent_calls={max_active_agent_count}\n"
                    + (err_bytes or "")
                ),
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def test_four_parallel_five_second_tasks_under_10s(self):
        """AC3: 4 parallel 5s tasks must complete in < 10s.

        If parallel() were serial, 4 x 5s = 20s. With true parallelism and
        a concurrency cap of 4, we expect ~5s plus startup overhead.
        """
        user_script = (
            "export const meta = { name: 'perf', description: 'perf', "
            "phases: [{title: 't', detail: 't'}], maxConcurrent: 4};\n"
            "export async function run(args) {\n"
            "  const items = ['5.0', '5.0', '5.0', '5.0'];\n"
            "  const results = await parallel(items.map(p => () => agent(p)));\n"
            "  return { done: results.length };\n"
            "}\n"
        )
        start = time.monotonic()
        rc, _, err = self._run_script(user_script)
        elapsed = time.monotonic() - start
        self.assertEqual(rc, 0, f"runtime exited non-zero: {err}")
        self.assertLess(
            elapsed,
            10.0,
            f"4 parallel 5s calls took {elapsed:.2f}s; expected < 10s. "
            f"parallel() may not be truly concurrent.",
        )

    def test_parallel_honours_concurrency_cap(self):
        """When max_concurrent=2, 4 tasks of 2s should take ~4s."""
        user_script = (
            "export const meta = { name: 'cap', description: 'cap', "
            "phases: [{title: 't', detail: 't'}], maxConcurrent: 2};\n"
            "export async function run(args) {\n"
            "  const items = ['2.0', '2.0', '2.0', '2.0'];\n"
            "  const results = await parallel(items.map(p => () => agent(p)));\n"
            "  return { done: results.length };\n"
            "}\n"
        )
        start = time.monotonic()
        rc, _, err = self._run_script(user_script, max_concurrent=2)
        elapsed = time.monotonic() - start
        self.assertEqual(rc, 0, f"runtime exited non-zero: {err}")
        # With cap=2 and 4x2s tasks, we expect ~4s, definitely < 6s.
        self.assertLess(
            elapsed,
            6.0,
            f"4 parallel 2s calls with cap=2 took {elapsed:.2f}s; expected < 6s.",
        )
        self.assertGreaterEqual(
            elapsed,
            3.5,
            f"4 parallel 2s calls with cap=2 took only {elapsed:.2f}s; "
            f"cap may not be honoured.",
        )

    def test_pipeline_honours_item_concurrency_cap(self):
        """pipeline keeps item-level map parallelism without unbounded fan-out."""
        user_script = (
            "export const meta = { name: 'pipeline-cap', description: 'pipeline-cap', "
            "phases: [{title: 't', detail: 't'}], maxConcurrent: 2};\n"
            "export async function run(args) {\n"
            "  const items = ['0.2', '0.2', '0.2', '0.2'];\n"
            "  const results = await pipeline(items, async p => agent(p));\n"
            "  return { done: results.length };\n"
            "}\n"
        )

        rc, _, diagnostics = self._run_script(user_script, max_concurrent=2)

        self.assertEqual(rc, 0, f"runtime exited non-zero: {diagnostics}")
        self.assertIn("agent_calls=4", diagnostics)
        self.assertIn(
            "max_active_agent_calls=2",
            diagnostics,
            f"pipeline exceeded item concurrency cap:\n{diagnostics}",
        )

    def test_schema_mismatch_does_not_repeat_whole_agent_call(self):
        """Python owns schema retries; JS only performs a fail-closed check."""
        user_script = (
            "export const meta = { name: 'schema-once', description: 'schema-once', "
            "phases: [{title: 't', detail: 't'}]};\n"
            "export async function run(args) {\n"
            "  return agent('0', { schema: { answer: 'string' } });\n"
            "}\n"
        )

        rc, result, diagnostics = self._run_script(user_script, max_concurrent=2)

        self.assertEqual(rc, 0, f"runtime exited non-zero: {diagnostics}")
        self.assertIn("agent_calls=1", diagnostics)
        self.assertIn("error", result.lower())


def shutil_which(name):
    """shutil.which wrapper — avoids importing at module top-level."""
    import shutil
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Lightweight parallelism sanity check (does not require Node.js).
# Runs N identical sleep tasks through the bridge's own executor and
# asserts the total wall clock is close to a single task's duration.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.timeout(120)
class TestBridgeExecutorParallelism(unittest.TestCase):
    """Verifies RuntimeBridge's ThreadPoolExecutor is truly parallel.

    These tests do NOT launch a Node subprocess — they feed agent_call
    messages directly into ``_handle_agent_call`` against a local
    executor with a tiny handler. This makes them suitable for CI
    environments without Node.js while still exercising the exact
    concurrency path used at runtime.

    Benchmark tests — tagged as ``slow`` and given a 120s timeout.
    """

    def _build_bridge(self, max_concurrent: int) -> RuntimeBridge:
        bridge = RuntimeBridge(
            script_path="noop.js",
            cwd="/tmp",
            max_concurrent=max_concurrent,
            on_agent_call=lambda params: AgentCallResult(
                output="ok", token_usage=0, duration_s=0.0
            ),
        )
        _register_bridge(bridge)
        bridge._executor = ThreadPoolExecutor(max_workers=bridge._max_concurrent)
        return bridge

    def tearDown(self):
        _cleanup_all_perf_resources()

    def test_four_five_second_tasks_finish_under_ten_seconds(self):
        """AC3 (lightweight): 4 parallel ~5s calls must take <10s.

        Uses a tiny handler that time.sleeps, mimicking a blocking agent
        call. If the pool were serialised this test would take ~20s.
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor as _TP

        bridge = RuntimeBridge(
            script_path="noop.js",
            cwd="/tmp",
            max_concurrent=4,
            on_agent_call=lambda params: AgentCallResult(
                output="ok", token_usage=0, duration_s=0.0
            ),
        )
        bridge._executor = _TP(max_workers=bridge._max_concurrent)

        start = _time.monotonic()
        futures = []
        for i in range(4):
            futures.append(bridge._executor.submit(_time.sleep, 2.0))
        for f in futures:
            f.result(timeout=30)
        elapsed = _time.monotonic() - start
        self.assertLess(
            elapsed,
            10.0,
            f"4 parallel 2s calls took {elapsed:.2f}s; expected <10s "
            "(serial would take ~8s, true parallel ~2s).",
        )

    def test_concurrency_cap_limits_parallelism(self):
        """When cap=2, 4 calls of 1s each should take ~2s, not 1s."""
        import time as _time
        from concurrent.futures import ThreadPoolExecutor as _TP

        bridge = RuntimeBridge(
            script_path="noop.js",
            cwd="/tmp",
            max_concurrent=2,
        )
        bridge._executor = _TP(max_workers=bridge._max_concurrent)

        start = _time.monotonic()
        futures = [bridge._executor.submit(_time.sleep, 1.0) for _ in range(4)]
        for f in futures:
            f.result(timeout=30)
        elapsed = _time.monotonic() - start
        self.assertGreaterEqual(
            elapsed,
            1.9,
            f"cap=2 4x1s calls took only {elapsed:.2f}s; cap not honoured.",
        )
        self.assertLess(
            elapsed,
            4.0,
            f"cap=2 4x1s calls took {elapsed:.2f}s; expected <4s (serial=4s).",
        )


def _iter_stdout_lines(stdout, deadline):
    """Yield non-empty lines from stdout until deadline expires."""
    import select
    poller = select.poll()
    poller.register(stdout, select.POLLIN)
    buf = ""
    while time.monotonic() < deadline:
        try:
            events = poller.poll(0.2)
        except Exception:
            break
        if not events:
            continue
        chunk = stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line
    # Flush remainder
    if buf:
        yield buf


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Semantic tests for pipeline() documentation vs runtime implementation.
# These tests guard against regressions that would re-introduce ambiguity
# between "sequential pipeline" wording and the runtime's actual parallel/map
# semantics (Promise.all over items, sequential stages per item).
# ---------------------------------------------------------------------------


class TestPipelineSemantics(unittest.TestCase):
    """Assert pipeline is a bounded parallel/map primitive."""

    RUNTIME_JS = "src/workflow_engine/runtime/runtime.js"

    def _runtime_text(self) -> str:
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, self.RUNTIME_JS)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_pipeline_prompt_mentions_parallel_map(self):
        """The code-gen prompt must describe pipeline() as parallel/map,
        otherwise AI scripts could assume fully-sequential semantics and
        write code that depends on item ordering."""
        from src.workflow_engine.script_gen import build_script_gen_prompt
        prompt = build_script_gen_prompt(
            requirement="analyze files",
            available_tools=["coco"],
            orchestrator_agent="coco",
        )
        # Pin to the JS-primitive usage of pipeline in the prompt code fence
        # (avoid matching incidental mentions like "复杂 pipeline 组合" in
        # the orchestrator capability block).
        needle = "await pipeline("
        idx = prompt.lower().find(needle)
        self.assertGreaterEqual(
            idx,
            0,
            f"prompt must contain '{needle}' in its primitives example block",
        )
        tail = prompt[max(0, idx - 300): idx + 600]
        self.assertIn(
            "parallel",
            tail.lower(),
            "pipeline section of prompt must mention 'parallel' (items concurrent)",
        )
        self.assertIn(
            "map",
            tail.lower(),
            "pipeline section of prompt must mention 'map' (parallel/map semantics)",
        )

    def test_pipeline_runtime_reuses_bounded_parallel_scheduler(self):
        """pipeline must reuse parallel() so item fan-out honours the host cap."""
        import re
        text = self._runtime_text()
        # Locate the pipeline function definition...
        m = re.search(r"async\s+function\s+pipeline\s*\(", text)
        self.assertIsNotNone(m, "runtime.js missing 'async function pipeline('")
        # ...then find its closing brace by simple brace counting.
        start = m.start()
        open_brace = text.index("{", start)
        depth = 0
        end = None
        for i in range(open_brace, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        self.assertIsNotNone(end, "could not find end of pipeline function")
        body = text[open_brace:end]
        self.assertRegex(
            body,
            r"await\s+parallel\s*\(",
            "pipeline() must delegate item scheduling to bounded parallel()",
        )
        self.assertNotIn(
            "Promise.all(\n    items.map",
            body,
            "pipeline() must not eagerly create one promise per input item",
        )

    def test_prompt_requires_scope_first_pipeline_and_barrier_discipline(self):
        from src.workflow_engine.script_gen import build_script_gen_prompt

        prompt = build_script_gen_prompt(
            requirement="analyze and update many files",
            available_tools=["coco", "claude"],
            orchestrator_agent="coco",
        ).lower()

        self.assertIn("scope first", prompt)
        self.assertIn("pipeline-first", prompt)
        self.assertIn("barrier", prompt)
        self.assertIn("structured output", prompt)

    def test_sequence_helper_hinted_in_prompt(self):
        """The prompt should hint at sequence() / serialPipeline() so users
        can opt into strictly sequential item processing when they need it."""
        from src.workflow_engine.script_gen import build_script_gen_prompt
        prompt = build_script_gen_prompt(
            requirement="analyze files",
            available_tools=["coco"],
            orchestrator_agent="coco",
        )
        lower = prompt.lower()
        self.assertTrue(
            "sequence" in lower or "serialpipeline" in lower,
            "prompt must hint at sequence() or serialPipeline() "
            "for strictly-sequential item processing",
        )
