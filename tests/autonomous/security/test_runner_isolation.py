"""Security tests: Runner isolation, worker sandboxing, oracle timeouts."""

import tempfile

import pytest

from src.autonomous.runtime.runner import SandboxRunner
from src.autonomous.runtime.worker import APPROVED_MODULES, execute_task
from src.autonomous.verifier.oracle_runner import OracleRunner

# ---------------------------------------------------------------------------
# SandboxRunner isolation tests
# ---------------------------------------------------------------------------


class TestSandboxRunnerIsolation:
    """SandboxRunner provides process-group isolation and timeout kill."""

    def test_probe_returns_status(self) -> None:
        runner = SandboxRunner()
        info = runner.probe()
        assert "bwrap_available" in info
        assert "process_group_available" in info
        assert info["process_group_available"] is True

    def test_simple_command_execution(self) -> None:
        runner = SandboxRunner()
        result = runner.run(["echo", "hello"], timeout=5.0)
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert result.timed_out is False

    @pytest.mark.slow
    def test_timeout_kills_process(self) -> None:
        runner = SandboxRunner()
        result = runner.run(["sleep", "60"], timeout=1.0)
        assert result.timed_out is True
        assert result.elapsed_seconds < 5.0  # Should not wait full 60s

    @pytest.mark.slow
    def test_timeout_kills_child_processes(self) -> None:
        """Process-group kill ensures descendants are also terminated."""
        runner = SandboxRunner()
        # Fork a child that also sleeps
        script = "sleep 60 & sleep 60 & wait"
        result = runner.run(["sh", "-c", script], timeout=1.0)
        assert result.timed_out is True

    def test_nonexistent_command_fails(self) -> None:
        runner = SandboxRunner()
        result = runner.run(["/nonexistent/binary_xyz"], timeout=5.0)
        assert result.returncode != 0 or "Failed to start" in result.stderr

    def test_stdin_data_passed(self) -> None:
        runner = SandboxRunner()
        result = runner.run(
            ["cat"],
            timeout=5.0,
            stdin_data=b"input data here",
        )
        assert result.returncode == 0
        assert "input data here" in result.stdout

    def test_minimal_environment(self) -> None:
        """When no env provided, uses minimal safe environment."""
        runner = SandboxRunner()
        result = runner.run(["env"], timeout=5.0)
        assert result.returncode == 0
        # Should have PATH but not inherit all parent env
        assert "PATH=" in result.stdout

    def test_custom_environment(self) -> None:
        runner = SandboxRunner()
        result = runner.run(
            ["sh", "-c", "echo $MY_VAR"],
            timeout=5.0,
            env={"PATH": "/usr/bin:/bin", "MY_VAR": "secret_value"},
        )
        assert "secret_value" in result.stdout

    def test_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runner = SandboxRunner()
            result = runner.run(["pwd"], timeout=5.0, cwd=td)
            assert td in result.stdout

    def test_empty_argv_rejected(self) -> None:
        runner = SandboxRunner()
        result = runner.run([], timeout=5.0)
        assert result.returncode == -1
        assert "Empty argv" in result.stderr


# ---------------------------------------------------------------------------
# Worker isolation tests
# ---------------------------------------------------------------------------


class TestWorkerIsolation:
    """Worker entrypoint restricts imports and execution scope."""

    def test_approved_modules_are_stdlib_only(self) -> None:
        """APPROVED_MODULES must contain only stdlib modules."""
        forbidden_prefixes = ["src.", "autonomous", "broker", "journal"]
        for mod in APPROVED_MODULES:
            for prefix in forbidden_prefixes:
                assert not mod.startswith(prefix), f"Non-stdlib module in APPROVED: {mod}"

    def test_eval_blocks_dunder_access(self) -> None:
        result = execute_task({
            "task_type": "eval",
            "payload": {"expression": "__import__('os').system('id')"},
        })
        assert result["success"] is False
        assert "restricted" in result["error"].lower() or "Forbidden" in result["error"]

    def test_eval_blocks_import_keyword(self) -> None:
        result = execute_task({
            "task_type": "eval",
            "payload": {"expression": "import os"},
        })
        assert result["success"] is False

    def test_eval_allows_safe_expressions(self) -> None:
        result = execute_task({
            "task_type": "eval",
            "payload": {"expression": "2 + 2"},
        })
        assert result["success"] is True
        assert result["output"] == 4

    def test_eval_with_context(self) -> None:
        result = execute_task({
            "task_type": "eval",
            "payload": {"expression": "x * 2", "context": {"x": 21}},
        })
        assert result["success"] is True
        assert result["output"] == 42

    def test_transform_keys(self) -> None:
        result = execute_task({
            "task_type": "transform",
            "payload": {"data": {"a": 1, "b": 2}, "operation": "keys"},
        })
        assert result["success"] is True
        assert sorted(result["output"]) == ["a", "b"]

    def test_transform_length(self) -> None:
        result = execute_task({
            "task_type": "transform",
            "payload": {"data": [1, 2, 3, 4], "operation": "length"},
        })
        assert result["success"] is True
        assert result["output"] == 4

    def test_unknown_task_type_rejected(self) -> None:
        result = execute_task({
            "task_type": "exec_arbitrary_code",
            "payload": {},
        })
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_worker_does_not_import_autonomous_internals(self) -> None:
        """Worker module must not import journal, broker, domain etc."""
        import src.autonomous.runtime.worker as worker_mod

        # Get all names defined in worker
        worker_source = open(worker_mod.__file__).read()
        # Must not contain imports from parent packages
        assert "from ..broker" not in worker_source
        assert "from ..domain" not in worker_source
        assert "from ..journal" not in worker_source
        assert "from ..policy" not in worker_source


# ---------------------------------------------------------------------------
# OracleRunner isolation tests
# ---------------------------------------------------------------------------


class TestOracleRunnerIsolation:
    """OracleRunner provides deterministic verification with timeout kill."""

    def test_oracle_probe(self) -> None:
        runner = OracleRunner(timeout=30.0)
        info = runner.probe()
        assert "oracle_timeout" in info
        assert info["oracle_timeout"] == 30.0

    def test_oracle_passing_command(self) -> None:
        runner = OracleRunner(timeout=10.0)
        result = runner.run_oracle(
            criterion_id="crit_1",
            command="echo 'all good' && exit 0",
        )
        assert result.passed is True
        assert result.exit_code == 0
        assert "all good" in result.stdout
        assert result.output_hash != ""

    def test_oracle_failing_command(self) -> None:
        runner = OracleRunner(timeout=10.0)
        result = runner.run_oracle(
            criterion_id="crit_2",
            command="echo 'FAIL' >&2 && exit 1",
        )
        assert result.passed is False
        assert result.exit_code == 1

    @pytest.mark.slow
    def test_oracle_timeout_kills_all_descendants(self) -> None:
        """Oracle timeout MUST kill entire process group."""
        runner = OracleRunner(timeout=1.0)
        result = runner.run_oracle(
            criterion_id="crit_timeout",
            command="sleep 60 & sleep 60 & wait",
        )
        assert result.timed_out is True
        assert result.passed is False
        assert result.elapsed_seconds < 5.0

    def test_oracle_empty_command_rejected(self) -> None:
        runner = OracleRunner(timeout=10.0)
        result = runner.run_oracle(criterion_id="crit_empty", command="")
        assert result.passed is False
        assert "Empty" in result.error

    def test_oracle_batch_execution(self) -> None:
        runner = OracleRunner(timeout=10.0)
        oracles = [
            {"criterion_id": "c1", "command": "exit 0"},
            {"criterion_id": "c2", "command": "exit 1"},
            {"criterion_id": "c3", "command": "echo ok"},
        ]
        results = runner.run_batch(oracles)
        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is False
        assert results[2].passed is True

    def test_oracle_output_hash_deterministic(self) -> None:
        """Same command produces same output hash."""
        runner = OracleRunner(timeout=10.0)
        r1 = runner.run_oracle(criterion_id="c1", command="echo deterministic")
        r2 = runner.run_oracle(criterion_id="c1", command="echo deterministic")
        assert r1.output_hash == r2.output_hash

    def test_oracle_does_not_access_journal(self) -> None:
        """OracleRunner source must not import journal/broker/domain state."""
        import src.autonomous.verifier.oracle_runner as oracle_mod
        source = open(oracle_mod.__file__).read()
        assert "from ..journal" not in source
        assert "from ..broker.model_broker" not in source
        assert "from ..broker.tool_broker" not in source
