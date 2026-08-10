"""Tests for ButtonIntent → action_id mapping completeness and correctness."""


from src.card.render.buttons import INTENT_TO_ACTION_ID
from src.card.state.button_intent import ButtonIntent


class TestButtonMappingCompleteness:
    """Verify all ButtonIntents have a mapping."""

    def test_all_intents_mapped(self):
        """Every ButtonIntent enum member must have an entry in INTENT_TO_ACTION_ID."""
        missing = []
        for intent in ButtonIntent:
            if intent.value not in INTENT_TO_ACTION_ID:
                missing.append(intent.name)
        assert not missing, f"Missing mappings for: {missing}"

    def test_mapping_values_are_strings(self):
        """All mapped action_ids are non-empty strings."""
        for intent_val, action_id in INTENT_TO_ACTION_ID.items():
            assert isinstance(action_id, str), f"action_id for {intent_val} is not a string"
            assert action_id, f"action_id for {intent_val} is empty"

    def test_mapping_keys_start_with_intent_prefix(self):
        """All mapping keys start with 'intent.' prefix."""
        for key in INTENT_TO_ACTION_ID:
            assert key.startswith("intent."), f"Key {key} doesn't start with 'intent.'"


class TestResolveActionIdFallback:
    """Unit tests for _resolve_action_id graceful degradation."""

    def test_known_intent_resolves(self):
        """Known ButtonIntent resolves to a non-empty action_id."""
        from src.card.render.buttons import _resolve_action_id
        from src.card.state.models import ButtonSpec

        spec = ButtonSpec(text="停止", action_id=ButtonIntent.SPEC_STOP.value)
        resolved = _resolve_action_id(spec)
        assert resolved is not None
        assert resolved == INTENT_TO_ACTION_ID[ButtonIntent.SPEC_STOP.value]

    def test_unknown_intent_returns_none(self):
        """Unknown ButtonIntent (starts with 'intent.') returns None."""
        from src.card.render.buttons import _resolve_action_id
        from src.card.state.models import ButtonSpec

        spec = ButtonSpec(text="???", action_id="intent.unknown.nonexistent")
        resolved = _resolve_action_id(spec)
        assert resolved is None

    def test_unknown_intent_logs_warning(self, caplog):
        """Unknown ButtonIntent logs a warning message."""
        import logging

        from src.card.render.buttons import _resolve_action_id
        from src.card.state.models import ButtonSpec

        spec = ButtonSpec(text="???", action_id="intent.bogus.action")
        with caplog.at_level(logging.WARNING, logger="src.card.render.buttons"):
            _resolve_action_id(spec)
        assert "Unknown ButtonIntent" in caplog.text
        assert "intent.bogus.action" in caplog.text

    def test_raw_action_id_passthrough(self):
        """Non-intent action_id (not starting with 'intent.') passes through as-is."""
        from src.card.render.buttons import _resolve_action_id
        from src.card.state.models import ButtonSpec

        spec = ButtonSpec(text="Custom", action_id="my_custom_action")
        resolved = _resolve_action_id(spec)
        assert resolved == "my_custom_action"

    def test_render_button_disabled_for_unknown_intent(self):
        """_render_button produces disabled button when intent is unresolved."""
        from src.card.render.buttons import _render_button
        from src.card.state.models import ButtonSpec

        spec = ButtonSpec(text="Missing", action_id="intent.does.not.exist")
        btn = _render_button(spec)
        assert btn["tag"] == "button"
        assert btn["disabled"] is True
        assert btn["text"]["content"] == "Missing"
        assert "value" not in btn
