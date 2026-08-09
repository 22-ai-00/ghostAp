"""Tests for the pure routing logic in resolve_command_route."""

from unittest.mock import MagicMock

import pytest

from src.feishu.dispatcher import MessageDispatcher
from src.feishu.request_context import RequestContext
from src.feishu.route_decision import RouteTarget


@pytest.fixture
def dispatcher():
    return MessageDispatcher(MagicMock())


class TestResolveCommandRoute:
    def test_deep_command(self, dispatcher):
        ctx = RequestContext(message_id="m1", chat_id="c1", text="/deep test task")
        decision = dispatcher.resolve_command_route(ctx)
        assert decision is not None
        assert decision.target == RouteTarget.DEEP_ENGINE

    def test_spec_command(self, dispatcher):
        ctx = RequestContext(message_id="m1", chat_id="c1", text="/spec build auth")
        decision = dispatcher.resolve_command_route(ctx)
        assert decision is not None
        assert decision.target == RouteTarget.SPEC_ENGINE

    def test_workflow_command(self, dispatcher):
        ctx = RequestContext(message_id="m1", chat_id="c1", text="/wf build pipeline")
        decision = dispatcher.resolve_command_route(ctx)
        assert decision is not None
        assert decision.target == RouteTarget.WORKFLOW_ENGINE

    def test_exit_command_in_programming(self, dispatcher):
        ctx = RequestContext(
            message_id="m1", chat_id="c1", text="/exit",
            is_in_programming=True,
        )
        decision = dispatcher.resolve_command_route(ctx)
        assert decision is not None
        assert decision.target == RouteTarget.EXIT_MODE

    def test_exit_command_not_in_programming(self, dispatcher):
        ctx = RequestContext(
            message_id="m1", chat_id="c1", text="/exit",
            is_in_programming=False,
        )
        decision = dispatcher.resolve_command_route(ctx)
        # Not in programming — /exit falls through to intent recognition
        assert decision is None

    def test_programming_mode_forwards(self, dispatcher):
        ctx = RequestContext(
            message_id="m1", chat_id="c1", text="fix the bug",
            is_in_programming=True,
        )
        decision = dispatcher.resolve_command_route(ctx)
        assert decision is not None
        assert decision.target == RouteTarget.PROGRAMMING_MODE

    def test_plain_text_returns_none(self, dispatcher):
        ctx = RequestContext(message_id="m1", chat_id="c1", text="hello world")
        decision = dispatcher.resolve_command_route(ctx)
        assert decision is None
