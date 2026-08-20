"""Core contracts for Spec objective verification."""

from unittest.mock import MagicMock, patch

from src.spec_engine.criteria import evaluate_criteria, run_objective_verify


def test_run_objective_verify_without_command_is_successful():
    assert run_objective_verify("", "/tmp") == (True, "")


def test_run_objective_verify_preserves_executor_outcome():
    for success, stream, expected in (
        (True, "5 passed", "5 passed"),
        (False, "2 failed", "2 failed"),
    ):
        result = MagicMock(success=success, stdout=stream if success else "", stderr="" if success else stream)
        executor = MagicMock()
        executor.return_value.execute.return_value = result
        with patch("src.command_executor.CommandExecutor", executor):
            passed, output = run_objective_verify("pytest tests/ -q", "/project")
        assert passed is success
        assert expected in output


def _emit_criteria(text):
    def send(_prompt, **kwargs):
        event = MagicMock(event_type="text_chunk", text=text)
        if callback := kwargs.get("on_event"):
            callback(event)

    return send


def _settings():
    return MagicMock(
        spec_objective_verify_enabled=True,
        spec_objective_verify_timeout=60,
        engine_eval_prompt_timeout=60,
    )


@patch("src.spec_engine.criteria.run_objective_verify", return_value=(False, "FAILED: 2 tests failed"))
def test_objective_verify_failure_overrides_llm_optimism(_verify):
    project = MagicMock(verify_command="pytest tests/ -q", root_path="/project")
    project.criteria_tracker.is_all_satisfied = True

    result = evaluate_criteria(
        session=MagicMock(),
        criteria=["test1", "test2"],
        cycle=1,
        project=project,
        send_prompt_fn=_emit_criteria("CRITERIA_1: PASS\nCRITERIA_2: PASS"),
        settings=_settings(),
    )

    assert result["all_satisfied"] is False
    assert result["verify_passed"] is False


def test_missing_verify_command_uses_llm_criteria_result():
    project = MagicMock(verify_command="", root_path="/project")
    project.criteria_tracker.is_all_satisfied = True

    result = evaluate_criteria(
        session=MagicMock(),
        criteria=["test1"],
        cycle=1,
        project=project,
        send_prompt_fn=_emit_criteria("CRITERIA_1: PASS"),
        settings=_settings(),
    )

    assert result["all_satisfied"] is True
    assert result["verify_passed"] is True
