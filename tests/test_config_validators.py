"""Tests for config.py validators: boundary values and Literal type constraints."""


import pytest
from pydantic import ValidationError

from src.config import CardSessionConfig, Settings


def _build_settings(**overrides) -> Settings:
    """Create a Settings instance with only the specified overrides.

    Uses model_validate to bypass env file reading.
    """
    defaults = {
        "app_id": "test",
        "app_secret": "test",
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


# ---------------------------------------------------------------------------
# Data-driven boundary specs
# (setting_key, attr_path, lower_bound, upper_bound, extra_kwargs_lower, extra_kwargs_upper)
# ---------------------------------------------------------------------------
_BOUNDARY_SPECS = [
    ("card_max_chars", "card.max_chars", 1000, 50000, {}, {}),
    ("card_session_lock_max", "card.session_lock_max", 1000, 100000, {}, {}),
    ("card_session_lock_ttl", "card.session_lock_ttl", 60, 3600, {}, {"card_session_idle_timeout": 7200}),
    ("card_session_idle_timeout", "card.session_idle_timeout", 300, 7200, {"card_session_lock_ttl": 60, "card_session_idle_warn_at_remaining": 60}, {}),
    ("card_action_dedup_ttl", "card.action_dedup_ttl", 0, 10, {}, {}),
    ("card_action_dedup_max_size", "card.action_dedup_max_size", 100, 50000, {}, {}),
    ("card_action_dedup_cleanup_interval", "card.action_dedup_cleanup_interval", 1, 3600, {}, {}),
    ("card_session_max_rotations", "card.session_max_rotations", 1, 100, {}, {}),
    ("card_delivery_pool_max_workers", "card.delivery_pool_max_workers", 1, 32, {}, {}),
    ("lock_undo_window_seconds", "lock_undo_window_seconds", 60, 600, {}, {}),
]
_BOUNDARY_IDS = [s[0] for s in _BOUNDARY_SPECS]


class TestBoundaryValidators:
    """Parametrized boundary tests for all range-validated settings."""

    @pytest.mark.parametrize("setting_key,attr_path,lower,upper,extra_lower,extra_upper", _BOUNDARY_SPECS, ids=_BOUNDARY_IDS)
    def test_below_lower_bound(self, setting_key, attr_path, lower, upper, extra_lower, extra_upper):
        with pytest.raises(ValidationError, match=setting_key):
            _build_settings(**{setting_key: lower - 1})

    @pytest.mark.parametrize("setting_key,attr_path,lower,upper,extra_lower,extra_upper", _BOUNDARY_SPECS, ids=_BOUNDARY_IDS)
    def test_above_upper_bound(self, setting_key, attr_path, lower, upper, extra_lower, extra_upper):
        with pytest.raises(ValidationError, match=setting_key):
            _build_settings(**{setting_key: upper + 1})


class TestBoundaryExtras:
    """Additional boundary checks not covered by the 4-point parametrized tests."""

    def test_action_dedup_ttl_mid_value(self):
        s = _build_settings(card_action_dedup_ttl=5)
        assert s.card.action_dedup_ttl == 5

    def test_action_dedup_max_size_default_in_range(self):
        s = _build_settings()
        assert 100 <= s.card.action_dedup_max_size <= 50000

    def test_lock_undo_window_default_value(self):
        s = _build_settings()
        assert s.lock_undo_window_seconds == 300

    def test_lock_undo_window_not_multiple_of_60_rejected(self):
        with pytest.raises(ValidationError, match="必须为 60 的整数倍"):
            _build_settings(lock_undo_window_seconds=90)

    def test_lock_undo_window_multiple_of_60_accepted(self):
        s = _build_settings(lock_undo_window_seconds=180)
        assert s.lock_undo_window_seconds == 180

    @pytest.mark.parametrize("setting_key,value", [
        ("card_session_max_rotations", -1),
        ("card_delivery_pool_max_workers", -1),
    ])
    def test_negative_values_rejected(self, setting_key, value):
        with pytest.raises(ValidationError, match=setting_key):
            _build_settings(**{setting_key: value})


class TestProgrammingTimeoutValidation:
    @pytest.mark.parametrize(
        "setting_key",
        ["coco_execution_timeout", "claude_execution_timeout"],
    )
    @pytest.mark.parametrize("value", [0, -1])
    def test_execution_timeout_must_be_positive(self, setting_key, value):
        with pytest.raises(ValidationError, match=setting_key):
            _build_settings(**{setting_key: value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_finalization_reserve_must_be_finite(self, value):
        with pytest.raises(ValidationError, match="programming_finalization_reserve_s"):
            _build_settings(programming_finalization_reserve_s=value)


class TestCardButtonLayoutLiteral:
    """Tests for card_button_layout Literal['desktop','mobile','responsive']."""

    def test_valid_values(self):
        for val in ("desktop", "mobile", "responsive"):
            s = _build_settings(card_button_layout=val)
            assert s.card.button_layout == val

    def test_invalid_value(self):
        with pytest.raises(ValidationError):
            _build_settings(card_button_layout="foobar")


class TestCardButtonSizeLiteral:
    """Tests for card_button_size Literal['small','medium','large']."""

    def test_valid_values(self):
        for val in ("small", "medium", "large"):
            s = _build_settings(card_button_size=val)
            assert s.card.button_size == val

    def test_invalid_value(self):
        with pytest.raises(ValidationError):
            _build_settings(card_button_size="big")


class TestSessionConfigBudgetClamp:
    """Verify SessionConfig clamps budget.visible_chars when > card_max_chars."""

    def test_budget_exceeds_card_max_chars_is_clamped(self, caplog):
        """When visible_chars > card_max_chars, factory clamps it during session creation."""
        from unittest.mock import MagicMock, patch

        from src.card.render.budget import RenderBudget
        from src.card.session.config import SessionCallbacks
        from src.card.session.factory import CardSessionFactory
        from src.card.state.models import CardMetadata

        mock_settings = MagicMock()
        mock_settings.card.max_chars = 20000
        mock_settings.card.button_size = "medium"
        mock_settings.card.session_idle_timeout = 1800
        mock_settings.card.session_idle_warn_at_remaining = 300

        oversized_budget = RenderBudget(visible_chars=25000)
        metadata = CardMetadata(engine_type="deep", mode_name="Test", mode_emoji="\U0001f535")

        delivery = MagicMock()
        factory = CardSessionFactory(delivery=delivery)
        cbs = SessionCallbacks(notify_callback=lambda _c, _t: None)

        with patch("src.config.get_settings", return_value=mock_settings):
            session = factory.create(chat_id="c1", metadata=metadata, budget=oversized_budget, callbacks=cbs)

        assert session._budget.visible_chars == 20000
        assert "clamping" in caplog.text.lower() or "exceeds" in caplog.text.lower()

    def test_budget_within_limit_unchanged(self):
        """When visible_chars <= card_max_chars, budget remains unchanged."""
        from unittest.mock import MagicMock, patch

        from src.card.render.budget import RenderBudget
        from src.card.session.config import SessionCallbacks
        from src.card.session.factory import CardSessionFactory
        from src.card.state.models import CardMetadata

        mock_settings = MagicMock()
        mock_settings.card.max_chars = 28000
        mock_settings.card.button_size = "medium"
        mock_settings.card.session_idle_timeout = 1800
        mock_settings.card.session_idle_warn_at_remaining = 300

        budget = RenderBudget(visible_chars=25000)
        metadata = CardMetadata(engine_type="deep", mode_name="Test", mode_emoji="\U0001f535")

        delivery = MagicMock()
        factory = CardSessionFactory(delivery=delivery)
        cbs = SessionCallbacks(notify_callback=lambda _c, _t: None)

        with patch("src.config.get_settings", return_value=mock_settings):
            session = factory.create(chat_id="c1", metadata=metadata, budget=budget, callbacks=cbs)

        assert session._budget.visible_chars == 25000


class TestCardSessionCrossFieldValidator:
    """Cross-field: card_session_lock_ttl must be <= card_session_idle_timeout."""

    def test_lock_ttl_greater_than_idle_timeout_raises(self):
        with pytest.raises(ValidationError, match="card_session_lock_ttl"):
            _build_settings(card_session_lock_ttl=3600, card_session_idle_timeout=300)

    def test_lock_ttl_equal_to_idle_timeout_accepted(self):
        s = _build_settings(card_session_lock_ttl=600, card_session_idle_timeout=600)
        assert s.card.session_lock_ttl == 600.0

    def test_default_values_pass(self):
        """Default 600 <= 1800 should pass."""
        s = _build_settings()
        assert s.card.session_lock_ttl <= s.card.session_idle_timeout


class TestValidatorMessagesContainUnit:
    """All card_session_* and lock_undo_window_seconds validators include '（秒）' in error."""

    @pytest.mark.parametrize("kwargs", [
        {"card_session_lock_ttl": 10},
        {"card_session_idle_timeout": 100},
        {"lock_undo_window_seconds": 30},
    ])
    def test_error_message_has_unit(self, kwargs):
        with pytest.raises(ValidationError, match=r"（秒）"):
            _build_settings(**kwargs)


class TestSessionIdleWarnCrossField:
    """session_idle_warn_at_remaining must be < session_idle_timeout."""

    def test_warn_equal_to_timeout_rejected(self):
        with pytest.raises(ValidationError, match="card_session_idle_warn_at_remaining"):
            _build_settings(
                card_session_idle_warn_at_remaining=1800,
                card_session_idle_timeout=1800,
                card_session_lock_ttl=60,
            )

    def test_warn_greater_than_timeout_rejected(self):
        with pytest.raises(ValidationError, match="card_session_idle_warn_at_remaining"):
            _build_settings(
                card_session_idle_warn_at_remaining=2000,
                card_session_idle_timeout=1800,
                card_session_lock_ttl=60,
            )

    def test_warn_less_than_timeout_accepted(self):
        s = _build_settings(
            card_session_idle_warn_at_remaining=1799,
            card_session_idle_timeout=1800,
            card_session_lock_ttl=60,
        )
        assert s.card.session_idle_warn_at_remaining == 1799


class TestEnvVarAliasResolution:
    """Both CARD_SESSION_IDLE_WARN_AT_REMAINING and CARD_SESSION_IDLE_WARN_BEFORE resolve to the same field."""

    def test_new_name_resolves(self):
        s = _build_settings(card_session_idle_warn_at_remaining=300)
        assert s.card.session_idle_warn_at_remaining == 300

    def test_old_name_resolves(self):
        s = _build_settings(card_session_idle_warn_before=300)
        assert s.card.session_idle_warn_at_remaining == 300

    def test_both_names_yield_same_value(self):
        s1 = _build_settings(card_session_idle_warn_at_remaining=250)
        s2 = _build_settings(card_session_idle_warn_before=250)
        assert s1.card.session_idle_warn_at_remaining == s2.card.session_idle_warn_at_remaining


# ---------------------------------------------------------------------------
# CardSessionConfig field validators (merged from test_config_validator.py)
# ---------------------------------------------------------------------------


class TestLockTTLAutoCeil:
    """Verify CARD_SESSION_LOCK_TTL auto-ceil to nearest 60 multiple."""

    def test_exact_multiple_unchanged(self):
        """60 should pass through unchanged."""
        config = CardSessionConfig(session_lock_ttl=60)
        assert config.session_lock_ttl == 60.0

    def test_90_ceils_to_120(self):
        """90 is not a multiple of 60 -> ceil to 120."""
        config = CardSessionConfig(session_lock_ttl=90)
        assert config.session_lock_ttl == 120.0


class TestLockTTLRangeRejection:
    """Verify out-of-range values raise ValueError."""

    def test_below_minimum_raises(self):
        """Values < 60 should raise ValueError."""
        with pytest.raises(ValueError, match="60.*3600"):
            CardSessionConfig(session_lock_ttl=30)

    def test_above_maximum_raises(self):
        """Values > 3600 should raise ValueError."""
        with pytest.raises(ValueError, match="60.*3600"):
            CardSessionConfig(session_lock_ttl=4000)
