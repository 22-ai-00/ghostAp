"""Workflow automatic-start configuration contract."""

from types import SimpleNamespace

from src.config.settings import Settings
from src.feishu.handlers.workflow import WorkflowHandler


def _handler(value: bool | None) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    settings = SimpleNamespace()
    if value is not None:
        settings.workflow_auto_execute = value
    handler.ctx = SimpleNamespace(settings=settings)
    return handler


def test_workflow_auto_execute_defaults_on() -> None:
    assert Settings(_env_file=None).workflow_auto_execute is True
    assert _handler(None)._auto_execute_workflow() is True


def test_workflow_auto_execute_can_be_disabled_by_settings() -> None:
    assert _handler(False)._auto_execute_workflow() is False


def test_workflow_auto_execute_reads_validated_environment(monkeypatch) -> None:
    monkeypatch.setenv("WORKFLOW_AUTO_EXECUTE", "false")
    settings = Settings(_env_file=None)
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = SimpleNamespace(settings=settings)

    assert settings.workflow_auto_execute is False
    assert handler._auto_execute_workflow() is False
