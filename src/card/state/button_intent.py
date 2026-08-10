"""ButtonIntent: Abstract button intents for all reducers.

Reducers emit ButtonIntent values instead of concrete action_id strings.
The render layer maps intents to action_ids via render/buttons.py.
"""

from __future__ import annotations

from enum import Enum


class ButtonIntent(str, Enum):
    """Abstract button intent identifiers used by reducers.

    These are decoupled from the concrete action_id strings used in
    Feishu card schemas. Mapping is handled by render/buttons.py.
    """
    # Engine control (shared across deep/spec)
    ENGINE_STOP = "intent.engine.stop"

    # Spec engine
    SPEC_STOP = "intent.spec.stop"

    # View mode toggle
    MODE_FULL = "intent.mode.full"
    MODE_COMPACT = "intent.mode.compact"

    # Global
    SHOW_STATUS = "intent.global.show_status"
