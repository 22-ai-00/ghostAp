"""Unit tests for BaseRenderer._build_hooks() hook injection strategy.

Verifies that the correct hooks are assembled based on caller arguments:
- context_update_fn provided (truthy) → (EmojiHook, ContextPersistenceHook)
- context_update_fn=None → (EmojiHook,) only

The injection *strategy* per engine (who passes context_update_fn) is:
- Deep engine: passes context_update_fn → gets ContextPersistenceHook
- Spec: does NOT pass context_update_fn → no ContextPersistenceHook
"""

from unittest.mock import MagicMock

import pytest

from src.card.hooks import ContextPersistenceHook, EmojiHook
from src.feishu.renderers.base import BaseRenderer


@pytest.fixture()
def mock_handler():
    """Minimal mock handler satisfying BaseRenderer constructor dependencies."""
    handler = MagicMock()
    handler.ctx = MagicMock()
    handler.settings = MagicMock()
    handler.add_reaction = MagicMock()
    handler.send_text_to_chat = MagicMock()
    return handler


@pytest.fixture()
def renderer(mock_handler):
    """BaseRenderer instance with mocked handler."""
    return BaseRenderer(mock_handler)


class TestBuildHooksWithContextFn:
    """When context_update_fn is provided, ContextPersistenceHook is injected."""

    @pytest.mark.parametrize(
        "engine_type",
        ["deep", "spec"],
        ids=["deep", "spec"],
    )
    def test_includes_persistence_hook(self, renderer, engine_type):
        """Any engine type with context_update_fn gets ContextPersistenceHook."""
        hooks = renderer._build_hooks(
            "msg_001",
            context_update_fn=lambda state: None,
            chat_id="chat_001",
            engine_type=engine_type,
        )
        assert len(hooks) == 2
        assert isinstance(hooks[0], EmojiHook)
        assert isinstance(hooks[1], ContextPersistenceHook)


class TestBuildHooksWithoutContextFn:
    """When context_update_fn is None, only EmojiHook is returned."""

    @pytest.mark.parametrize(
        "engine_type",
        ["deep", "spec"],
        ids=["deep", "spec"],
    )
    def test_only_emoji_hook(self, renderer, engine_type):
        """No ContextPersistenceHook without context_update_fn."""
        hooks = renderer._build_hooks(
            "msg_002",
            context_update_fn=None,
            chat_id="chat_001",
            engine_type=engine_type,
        )
        assert len(hooks) == 1
        assert isinstance(hooks[0], EmojiHook)
        assert not any(isinstance(h, ContextPersistenceHook) for h in hooks)


class TestBuildHooksInjectionStrategy:
    """Test the injection strategy matching actual engine caller patterns.

    Deep engine passes context_update_fn → gets ContextPersistenceHook.
    Spec does NOT pass context_update_fn → no ContextPersistenceHook.
    """

    def test_deep_engine_pattern(self, renderer):
        """Deep engine callers provide context_update_fn."""
        hooks = renderer._build_hooks(
            "msg_deep",
            context_update_fn=lambda state: None,
            chat_id="chat_deep",
            engine_type="deep",
        )
        assert len(hooks) == 2
        assert isinstance(hooks[0], EmojiHook)
        assert isinstance(hooks[1], ContextPersistenceHook)

    def test_spec_engine_pattern(self, renderer):
        """Spec engine callers do NOT provide context_update_fn."""
        hooks = renderer._build_hooks("msg_spec", chat_id="chat_spec", engine_type="spec")
        assert len(hooks) == 1
        assert isinstance(hooks[0], EmojiHook)

class TestBuildHooksWiring:
    """Verify hooks are wired correctly to handler methods."""

    def test_emoji_hook_wired_to_handler(self, renderer, mock_handler):
        hooks = renderer._build_hooks("msg_003", chat_id="chat_x")
        emoji_hook = hooks[0]
        assert emoji_hook._add_reaction is mock_handler.add_reaction
        assert emoji_hook._message_id == "msg_003"
        assert emoji_hook._chat_id == "chat_x"

    def test_persistence_hook_wired_to_handler(self, renderer, mock_handler):
        update_fn = MagicMock()
        hooks = renderer._build_hooks(
            "msg_004",
            context_update_fn=update_fn,
            chat_id="chat_y",
            engine_type="deep",
        )
        persistence_hook = hooks[1]
        assert isinstance(persistence_hook, ContextPersistenceHook)
        assert persistence_hook._update_fn is update_fn
        assert persistence_hook._notify_callback is mock_handler.send_text_to_chat
        assert persistence_hook._chat_id == "chat_y"
        assert persistence_hook._engine_type == "deep"
