"""Tests for src.card.engine_meta — centralized engine metadata."""

from src.card.engine_meta import (
    ENGINE_CMD_MAP,
    ENGINE_LABELS,
    ENGINE_NAME_MAP,
    engine_type_to_cmd,
    engine_type_to_name,
)


class TestEngineMeta:
    """Verify engine metadata consistency and helper behavior."""

    def test_all_maps_have_same_keys(self):
        """ENGINE_CMD_MAP, ENGINE_NAME_MAP, ENGINE_LABELS must cover the same engine types."""
        assert set(ENGINE_CMD_MAP.keys()) == set(ENGINE_NAME_MAP.keys())
        assert set(ENGINE_CMD_MAP.keys()) == set(ENGINE_LABELS.keys())

    def test_engine_type_to_cmd(self):
        """engine_type_to_cmd handles known/unknown types with default and custom fallbacks."""
        cases = (
            ("deep", None, "/deep"),
            ("spec", None, "/spec"),
            ("worktree", None, "/wt"),
            ("unknown", None, "命令"),
            ("", None, "命令"),
            (None, None, "命令"),
            ("unknown", "对应命令", "对应命令"),
            (None, "", ""),
        )
        for engine_type, fallback, expected in cases:
            actual = (
                engine_type_to_cmd(engine_type)
                if fallback is None
                else engine_type_to_cmd(engine_type, fallback=fallback)
            )
            assert actual == expected, (engine_type, fallback)

    def test_engine_type_to_name(self):
        """engine_type_to_name handles known/unknown types with default and custom fallbacks."""
        cases = (
            ("deep", None, "Deep"),
            ("spec", None, "Spec"),
            ("worktree", None, "Worktree"),
            ("unknown", None, ""),
            (None, None, ""),
            (None, "Engine", "Engine"),
        )
        for engine_type, fallback, expected in cases:
            actual = (
                engine_type_to_name(engine_type)
                if fallback is None
                else engine_type_to_name(engine_type, fallback=fallback)
            )
            assert actual == expected, (engine_type, fallback)

    def test_engine_labels_contain_restart_prefix(self):
        """All engine labels should contain the restart prefix."""
        for label in ENGINE_LABELS.values():
            assert "🔄" in label
