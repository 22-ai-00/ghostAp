from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, PropertyMock, patch

from src.feishu.handlers.programming import ClaudeModeHandler, Tui2acpModeHandler


class _FakeProgrammingCardSession:
    def __init__(self, *_args, **_kwargs):
        self.failed_text = None

    def start(self):
        return None

    def get_message_id(self):
        return None

    def wait_until_visible(self, _timeout):
        return True

    def abort(self):
        return None

    def wait_delivery_idle(self, _timeout):
        return True

    def terminal_delivery_succeeded(self):
        return True

    def on_event(self, _event):
        return None

    def fail(self, text, **_kwargs):
        self.failed_text = text


def _make_handler():
    ctx = MagicMock()
    ctx.settings = MagicMock()
    ctx.settings.claude_execution_timeout = 600
    ctx.settings.coco_execution_timeout = 600
    ctx.settings.repo_lock_hard_timeout = 3600
    ctx.api_client_factory = MagicMock()
    ctx.pending_image_lock = nullcontext()
    ctx.pending_image_keys = {}
    ctx.message_linker = MagicMock()
    ctx.context_manager = MagicMock()

    with patch.object(Tui2acpModeHandler, "settings", new_callable=PropertyMock, return_value=ctx.settings):
        handler = Tui2acpModeHandler.__new__(Tui2acpModeHandler)
        handler.ctx = ctx
        handler._settings = ctx.settings
        handler._current_adapter = None

    handler.mode_name = "Tui2ACP"
    handler.is_coco = False
    handler.reply_text = MagicMock()
    handler.add_reaction = MagicMock()
    handler.register_message_project = MagicMock()
    handler.ensure_request_id = MagicMock(return_value="req-1")
    handler._get_model_name_override = MagicMock(return_value=None)
    return handler


def _make_claude_handler():
    handler = _make_handler()
    handler.__class__ = ClaudeModeHandler
    handler.mode_name = "Claude"
    handler.interaction_mode = ClaudeModeHandler.interaction_mode
    return handler


def test_tui2acp_terminal_state_prompt_error_ends_manager_session():
    handler = _make_handler()
    manager = MagicMock()
    handler._get_session_manager = MagicMock(return_value=manager)

    session = MagicMock()
    session.session_id = "sid-1"
    session.message_count = 1
    session.send_prompt.side_effect = RuntimeError(
        "Session sid-1 is in terminal state"
    )

    with (
        patch("src.card.delivery.factory.create_card_delivery", return_value=MagicMock()),
        patch("src.card.delivery.feishu_client.FeishuCardAPIClient", return_value=MagicMock()),
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch("src.card.programming_adapter.ProgrammingCardSession", _FakeProgrammingCardSession),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    manager.end_session.assert_called_once_with(
        "chat-1",
        project_id=None,
        thread_id=None,
    )


def test_terminal_state_prompt_error_does_not_end_regular_acp_session():
    handler = _make_claude_handler()
    manager = MagicMock()
    handler._get_session_manager = MagicMock(return_value=manager)

    session = MagicMock()
    session.session_id = "sid-1"
    session.message_count = 1
    session.send_prompt.side_effect = RuntimeError(
        "Session sid-1 is in terminal state"
    )

    with (
        patch("src.card.delivery.factory.create_card_delivery", return_value=MagicMock()),
        patch("src.card.delivery.feishu_client.FeishuCardAPIClient", return_value=MagicMock()),
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch("src.card.programming_adapter.ProgrammingCardSession", _FakeProgrammingCardSession),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    manager.end_session.assert_not_called()


def test_programming_handle_response_builds_channel_card_client():
    handler = _make_handler()
    channel = object()
    handler.ctx.channel_client_factory = MagicMock(return_value=channel)
    session = MagicMock()
    session.session_id = "sid-1"
    session.message_count = 1
    session.send_prompt.side_effect = RuntimeError("stop after transport setup")
    channel_adapter = MagicMock(name="channel_card_adapter")
    delivery_factory = MagicMock(return_value=MagicMock())

    with (
        patch(
            "src.card.delivery.factory.create_card_delivery",
            delivery_factory,
        ),
        patch(
            "src.card.delivery.channel_client.LarkChannelCardAPIClient",
            return_value=channel_adapter,
        ) as channel_client_cls,
        patch(
            "src.card.delivery.feishu_client.FeishuCardAPIClient",
            return_value=MagicMock(name="legacy_card_adapter"),
        ),
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch("src.card.programming_adapter.ProgrammingCardSession", _FakeProgrammingCardSession),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    channel_client_cls.assert_called_once()
    assert channel_client_cls.call_args.args == (channel,)
    assert channel_client_cls.call_args.kwargs["preallocate_cards"] is True
    delivery_factory.assert_called_once_with(channel_adapter)


def test_programming_handle_response_falls_back_when_channel_is_unavailable():
    handler = _make_handler()
    handler.ctx.channel_client_factory = None
    handler._handle_response_non_streaming = MagicMock()
    session = MagicMock()

    handler.handle_response(
        "msg-1",
        "chat-1",
        "hello",
        session,
        None,
        "/tmp",
        "/tmp",
    )

    handler._handle_response_non_streaming.assert_called_once_with(
        "msg-1",
        "chat-1",
        "hello",
        session,
        None,
        "/tmp",
        _repo_lock_mgr=None,
        _root_path=None,
        _finalization_task_text=None,
    )
    session.send_prompt.assert_not_called()


def test_programming_falls_back_when_initial_async_channel_card_is_not_visible():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    handler._handle_response_non_streaming = MagicMock()
    session = MagicMock()

    class _InvisibleProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.aborted = False
            type(self).last = self

        def wait_until_visible(self, _timeout):
            return False

        def abort(self):
            self.aborted = True

    with (
        patch("src.card.delivery.factory.create_card_delivery", return_value=MagicMock()),
        patch(
            "src.card.delivery.channel_client.LarkChannelCardAPIClient",
            return_value=MagicMock(),
        ),
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch(
            "src.card.programming_adapter.ProgrammingCardSession",
            _InvisibleProgrammingCardSession,
        ),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    assert _InvisibleProgrammingCardSession.last.aborted is True
    handler._handle_response_non_streaming.assert_called_once()
    session.send_prompt.assert_not_called()


def test_programming_terminal_delivery_failure_aborts_retry_and_replies_text():
    handler = _make_handler()
    handler.ctx.channel_client_factory = MagicMock(return_value=object())
    session = MagicMock()
    session.session_id = "sid-1"
    session.message_count = 1
    session.send_prompt.side_effect = RuntimeError("terminal delivery test")

    class _FailedTerminalProgrammingCardSession(_FakeProgrammingCardSession):
        last = None

        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.aborted = False
            type(self).last = self

        def terminal_delivery_succeeded(self):
            return False

        def abort(self):
            self.aborted = True

    with (
        patch("src.card.delivery.factory.create_card_delivery", return_value=MagicMock()),
        patch(
            "src.card.delivery.channel_client.LarkChannelCardAPIClient",
            return_value=MagicMock(),
        ),
        patch("src.card.session.CardSession", return_value=MagicMock()),
        patch("src.card.session.factory.CardSessionFactory", return_value=MagicMock()),
        patch(
            "src.card.programming_adapter.ProgrammingCardSession",
            _FailedTerminalProgrammingCardSession,
        ),
    ):
        handler.handle_response(
            "msg-1",
            "chat-1",
            "hello",
            session,
            None,
            "/tmp",
            "/tmp",
        )

    assert _FailedTerminalProgrammingCardSession.last.aborted is True
    handler.reply_text.assert_called_once()
    assert handler.reply_text.call_args.args[0] == "msg-1"
    assert "terminal delivery test" in handler.reply_text.call_args.args[1]
