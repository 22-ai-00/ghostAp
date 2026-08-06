from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_startup_utils_exposes_safe_float_and_initial_diagnostics():
    from src.acp.startup_utils import initial_startup_diagnostics, safe_float_or_none

    error = RuntimeError("boom")

    assert safe_float_or_none("2.5") == 2.5
    assert safe_float_or_none("bad") is None

    diag = initial_startup_diagnostics(
        agent_type="coco",
        cwd="/tmp/project",
        model_name=None,
        error=error,
        attempt=1,
        retries=2,
        timeout_s="3.5",
    )

    assert diag["agent_type"] == "coco"
    assert diag["cwd"] == "/tmp/project"
    assert diag["error_type"] == "RuntimeError"
    assert diag["timeout_s"] == 3.5
    assert diag["args"] == []


def test_startup_collaborators_are_independently_testable():
    from src.acp.startup_utils import (
        StartupBackend,
        build_retry_plan,
        normalize_startup_cwd,
        select_startup_backend,
        should_try_injected_starter,
    )

    assert select_startup_backend("claude") == StartupBackend.CLI
    assert select_startup_backend("coco") == StartupBackend.ACP
    def starter(**kwargs):
        return None
    assert should_try_injected_starter("coco", starter=starter) is True

    assert normalize_startup_cwd(".", normalize_fn=lambda raw: "/repo") == "/repo"
    assert normalize_startup_cwd("/repo", normalize_fn=lambda raw: "") == "/repo"

    assert build_retry_plan("claude", retries=3, startup_timeout=10).attempts == 1
    acp_plan = build_retry_plan("coco", retries=2, startup_timeout=10)
    assert acp_plan.attempts == 2
    assert acp_plan.timeout_for_attempt(1) == 10
    assert acp_plan.timeout_for_attempt(2) == 15


def test_startup_error_classes_have_distinct_outcomes():
    from src.acp.startup_utils import (
        StartupErrorAction,
        StartupErrorKind,
        classify_startup_error,
    )

    injected = classify_startup_error(RuntimeError("injected failed"), phase="injected_starter", attempt=1, retries=2)
    assert injected.kind == StartupErrorKind.RECOVERABLE
    assert injected.action == StartupErrorAction.FALLBACK

    retryable = classify_startup_error(TimeoutError("handshake timeout"), phase="acp_retry", attempt=1, retries=2)
    assert retryable.kind == StartupErrorKind.RECOVERABLE
    assert retryable.action == StartupErrorAction.RETRY

    exhausted = classify_startup_error(TimeoutError("handshake timeout"), phase="acp_retry", attempt=2, retries=2)
    assert exhausted.kind == StartupErrorKind.DEGRADED
    assert exhausted.action == StartupErrorAction.REQUIRES_USER_ACTION

    fatal = classify_startup_error(RuntimeError("bad backend"), phase="unsupported_backend", attempt=1, retries=1)
    assert fatal.kind == StartupErrorKind.FATAL
    assert fatal.action == StartupErrorAction.RAISE


def test_startup_operational_exceptions_do_not_include_programming_errors():
    from src.acp.startup_utils import STARTUP_OPERATIONAL_EXCEPTIONS, StartupOperationalError

    assert Exception not in STARTUP_OPERATIONAL_EXCEPTIONS
    assert RuntimeError not in STARTUP_OPERATIONAL_EXCEPTIONS
    assert StartupOperationalError in STARTUP_OPERATIONAL_EXCEPTIONS
    assert TypeError not in STARTUP_OPERATIONAL_EXCEPTIONS
    assert AssertionError not in STARTUP_OPERATIONAL_EXCEPTIONS


def test_run_startup_operation_wraps_unknown_exception_as_fatal_not_operational():
    from src.acp.startup_utils import StartupFatalError, run_startup_operation

    class UnknownStartupBug(Exception):
        pass

    with pytest.raises(StartupFatalError) as exc_info:
        run_startup_operation(lambda: (_ for _ in ()).throw(UnknownStartupBug("bug")))

    assert isinstance(exc_info.value.__cause__, UnknownStartupBug)


def test_classify_startup_error_requires_user_action_for_exhausted_degrade():
    from src.acp.startup_utils import StartupErrorAction, StartupErrorKind, classify_startup_error

    exhausted = classify_startup_error(TimeoutError("handshake timeout"), phase="acp_retry", attempt=2, retries=2)

    assert exhausted.kind == StartupErrorKind.DEGRADED
    assert exhausted.action == StartupErrorAction.REQUIRES_USER_ACTION
    assert exhausted.user_reachable is True


def test_session_startup_unknown_exception_propagates_without_fallback_or_diagnostics():
    from src.acp.startup_utils import SessionStartupCoordinator, SessionStartupRequest

    class UnknownStartupBug(Exception):
        pass

    telemetry = MagicMock()

    def starter(**kwargs):
        raise UnknownStartupBug("programming bug must surface")

    coordinator = SessionStartupCoordinator(
        manager_agent_type="coco",
        session_starter=starter,
        session_telemetry=telemetry,
    )

    with pytest.raises(UnknownStartupBug):
        coordinator.start(
            SessionStartupRequest(
                key="chat",
                cwd="/tmp/project",
                startup_timeout=0.1,
                project_id=None,
                session_id=None,
                effective_agent_type="coco",
                model_name=None,
                retries=1,
            )
        )

    telemetry.on_session_start_failed.assert_not_called()


def test_session_startup_coordinator_is_split_into_startup_collaborators():
    """Final guard: ACP startup debt must not stay hidden in one broad method."""

    root = Path(__file__).resolve().parents[1]
    manager_source = (root / "src" / "acp" / "manager.py").read_text(encoding="utf-8")
    startup_source = (root / "src" / "acp" / "startup_utils.py").read_text(encoding="utf-8")
    memory = (root / ".Memory" / "2026-05-11.md").read_text(encoding="utf-8")
    backlog = (root / ".Memory" / "Backlog.md").read_text(encoding="utf-8")

    assert "class SessionStartupCoordinator" not in manager_source
    assert "class SessionStartupCoordinator" in startup_source
    assert "def start(self, request: SessionStartupRequest)" in startup_source
    for helper_name in (
        "InjectedStarterFallback",
        "AcpRetryStarter",
        "StartupFailureReporter",
        "select_startup_backend",
        "build_retry_plan",
        "build_manager_startup_diagnostics",
    ):
        assert helper_name in startup_source, f"SessionStartupCoordinator.start 未接入启动协作者 {helper_name}"
    assert "refactoring-analysis #4" not in backlog
    assert "| 4 | _start_session_inner God Function | 存在 |" in memory


def test_session_startup_coordinator_start_is_only_an_orchestrator():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src" / "acp" / "startup_utils.py").read_text(encoding="utf-8"))
    target: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SessionStartupCoordinator":
            target = next(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "start")
            break
    assert target is not None

    forbidden_call_names = {
        "build_manager_startup_diagnostics",
        "build_startup_diagnostics",
        "run_startup_operation",
        "SyncACPSession",
        "SyncClaudeCLISession",
    }
    calls: list[str] = []
    for call in (node for node in ast.walk(target) if isinstance(node, ast.Call)):
        func = call.func
        if isinstance(func, ast.Name):
            calls.append(func.id)
        elif isinstance(func, ast.Attribute):
            calls.append(func.attr)

    leaked = sorted(forbidden_call_names.intersection(calls))
    assert leaked == [], f"coordinator.start 泄漏启动细节调用: {leaked}"
    assert sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(target)) == 0


def test_session_startup_coordinator_delegates_backend_specific_branches_to_strategies():
    """Coordinator may select a backend, but backend startup behavior lives in strategies."""

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src" / "acp" / "startup_utils.py").read_text(encoding="utf-8"))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

    assert "StartupBackendStrategy" in class_names
    assert "AcpStartupBackendStrategy" in class_names
    assert "CliStartupBackendStrategy" in class_names

    coordinator = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "SessionStartupCoordinator"
    )
    coordinator_method_names = {
        node.name for node in coordinator.body if isinstance(node, ast.FunctionDef)
    }
    assert "_start_retry_backend_session" not in coordinator_method_names

    start_method = next(
        node for node in coordinator.body if isinstance(node, ast.FunctionDef) and node.name == "start"
    )
    backend_branch_lines = []
    for node in ast.walk(start_method):
        if not isinstance(node, ast.Compare):
            continue
        compared_names = []
        for item in [node.left, *node.comparators]:
            if isinstance(item, ast.Name):
                compared_names.append(item.id)
            elif isinstance(item, ast.Attribute):
                compared_names.append(item.attr)
        if "backend" in compared_names or any(name in {"ACP", "CLI"} for name in compared_names):
            backend_branch_lines.append(node.lineno)

    assert backend_branch_lines == [], f"coordinator.start 仍包含后端特定分支: {backend_branch_lines}"






def test_acp_retry_starter_retries_and_builds_diagnostics_without_success_side_effects():
    from src.acp.startup_utils import AcpRetryStarter, RetryPlan, StartupBackend

    class FlakySession:
        starts = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def describe_agent(self) -> str:
            return "flaky spec"

        def start(self, *, startup_timeout: float) -> str:
            FlakySession.starts += 1
            raise TimeoutError(f"timeout-{FlakySession.starts}-{startup_timeout}")

        def close(self) -> None:
            self.closed = True

    diagnostics: list[dict[str, object]] = []

    def diagnostics_fn(**kwargs):
        diagnostics.append(kwargs)
        return {"error_text": "handshake timed out", "attempts": []}

    result = AcpRetryStarter().start(
        backend=StartupBackend.ACP,
        agent_type="coco",
        cwd="/repo",
        model_name="model-a",
        retry_plan=RetryPlan(attempts=2, startup_timeout=10),
        request_key="chat/project",
        acp_session_cls=FlakySession,
        cli_session_cls=None,
        start_operation=lambda fn, **kwargs: fn(**kwargs),
        diagnostics_fn=diagnostics_fn,
        format_log_line_fn=lambda **kwargs: "failed",
        get_settings_fn=lambda: object(),
        sleep_fn=lambda seconds: None,
    )

    assert result.session is None
    assert result.actual_id == ""
    assert isinstance(result.last_err, TimeoutError)
    assert FlakySession.starts == 2
    assert [item["attempt"] for item in diagnostics] == [1, 2]


def test_acp_retry_starter_shares_one_absolute_deadline_across_retries():
    from src.acp.startup_utils import AcpRetryStarter, RetryPlan, StartupBackend

    clock = {"now": 100.0}
    attempt_timeouts: list[float] = []
    sleeps: list[float] = []

    class TimedSession:
        def __init__(self, **_kwargs):
            self.closed = False

        def describe_agent(self) -> str:
            return "timed spec"

        def start(self, *, startup_timeout: float) -> str:
            attempt_timeouts.append(startup_timeout)
            clock["now"] += 4.0 if len(attempt_timeouts) == 1 else startup_timeout
            raise TimeoutError("attempt failed")

        def close(self) -> None:
            self.closed = True

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    result = AcpRetryStarter().start(
        backend=StartupBackend.ACP,
        agent_type="coco",
        cwd="/repo",
        model_name=None,
        retry_plan=RetryPlan(attempts=3, startup_timeout=10),
        request_key="chat/project",
        acp_session_cls=TimedSession,
        cli_session_cls=None,
        start_operation=lambda fn, **kwargs: fn(**kwargs),
        diagnostics_fn=lambda **_kwargs: {},
        format_log_line_fn=lambda **_kwargs: "failed",
        get_settings_fn=lambda: object(),
        sleep_fn=sleep,
        deadline_monotonic=110.0,
        monotonic_fn=lambda: clock["now"],
    )

    assert result.session is None
    assert isinstance(result.last_err, TimeoutError)
    assert attempt_timeouts == pytest.approx([10.0, 5.7])
    assert sleeps == pytest.approx([0.3])
    assert clock["now"] == pytest.approx(110.0)


def test_session_startup_coordinator_retires_injected_late_success():
    from src.acp.startup_utils import SessionStartupCoordinator, SessionStartupRequest

    clock = {"now": 100.0}

    class LateSession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    late_session = LateSession()

    def starter(**_kwargs):
        clock["now"] = 111.0
        return late_session, "late-session", {}

    coordinator = SessionStartupCoordinator(
        manager_agent_type="coco",
        session_starter=starter,
        session_telemetry=MagicMock(),
        monotonic_fn=lambda: clock["now"],
    )

    with pytest.raises(TimeoutError, match="deadline"):
        coordinator.start(
            SessionStartupRequest(
                key="chat",
                cwd="/repo",
                startup_timeout=10,
                project_id=None,
                session_id=None,
                effective_agent_type="coco",
                model_name=None,
                retries=2,
                deadline_monotonic=110.0,
            )
        )

    assert late_session.closed is True


def test_session_startup_coordinator_has_no_uncategorized_broad_catches():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src" / "acp" / "startup_utils.py").read_text(encoding="utf-8"))
    target: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SessionStartupCoordinator":
            target = next(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "start"
            )
            break
    assert target is not None

    broad_handlers = [
        handler.lineno
        for handler in ast.walk(target)
        if isinstance(handler, ast.ExceptHandler)
        and isinstance(handler.type, ast.Name)
        and handler.type.id == "Exception"
    ]
    assert broad_handlers == []
