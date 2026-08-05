"""Task 19: Verify _BLOCK_KIND_MAP and _get_block_kind_handlers stay in sync."""


def test_block_kind_map_consistent_with_handlers():
    """Registries are consistent.

    This test verifies programmatically that models._BLOCK_KIND_MAP (minus
    tool_call which is handled by lookahead logic) matches exactly
    _get_block_kind_handlers() keys.
    """
    from src.card.render.atoms import _get_block_kind_handlers
    from src.card.state.models import _BLOCK_KIND_MAP

    model_keys = set(_BLOCK_KIND_MAP.keys()) - {"tool_call"}
    handler_keys = set(_get_block_kind_handlers().keys())
    assert model_keys == handler_keys, (
        f"Mismatch:\n"
        f"  In models but not handlers: {model_keys - handler_keys}\n"
        f"  In handlers but not models: {handler_keys - model_keys}"
    )
