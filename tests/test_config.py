"""Tests for config module singleton management functions."""

import pytest

from src.config import Settings, _reset_settings_for_testing, get_settings, set_settings


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure config singleton is reset before each test."""
    _reset_settings_for_testing()
    yield
    _reset_settings_for_testing()


def test_set_settings_updates_global_singleton():
    """Verify that set_settings correctly replaces the global settings singleton."""
    # Create a mock settings instance
    mock_settings = Settings()
    mock_settings.app_id = "test_app_id_set"  # type: ignore

    # First, get the default settings
    original = get_settings()
    assert original is not mock_settings

    # Now set our mock
    set_settings(mock_settings)

    # Verify that subsequent get_settings returns our mock
    retrieved = get_settings()
    assert retrieved is mock_settings
    assert retrieved.app_id == "test_app_id_set"  # type: ignore


def test_set_settings_respects_thread_safety():
    """Verify that set_settings works correctly with thread safety."""
    # This test ensures the lock is properly used (but doesn't test race conditions)
    mock1 = Settings()
    mock2 = Settings()

    set_settings(mock1)
    assert get_settings() is mock1

    set_settings(mock2)
    assert get_settings() is mock2


def test_set_settings_docstring():
    """Verify set_settings has appropriate docstring."""
    import inspect
    doc = inspect.getdoc(set_settings)
    assert doc is not None
    assert "dependency injection" in doc.lower() or "testing" in doc.lower()


# ---------------------------------------------------------------------------
# field_validator boundary tests — repo_lock_idle_timeout / cleanup_interval
# ---------------------------------------------------------------------------

class TestRepoLockTimerValidation:
    """Verify that repo_lock_idle_timeout, repo_lock_cleanup_interval, and
    repo_lock_hard_timeout reject non-positive values (AC-R7 / AC-R09)."""

    def test_idle_timeout_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_idle_timeout=0)
        assert "repo_lock_idle_timeout" in str(exc_info.value)

    def test_idle_timeout_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_idle_timeout=-1)
        assert "repo_lock_idle_timeout" in str(exc_info.value)

    def test_idle_timeout_one_ok(self):
        s = Settings(repo_lock_idle_timeout=1)
        assert s.repo_lock_idle_timeout == 1

    def test_cleanup_interval_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_cleanup_interval=0)
        assert "repo_lock_cleanup_interval" in str(exc_info.value)

    def test_cleanup_interval_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_cleanup_interval=-1)
        assert "repo_lock_cleanup_interval" in str(exc_info.value)

    def test_cleanup_interval_one_ok(self):
        s = Settings(repo_lock_cleanup_interval=1)
        assert s.repo_lock_cleanup_interval == 1

    def test_hard_timeout_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_hard_timeout=0)
        assert "repo_lock_hard_timeout" in str(exc_info.value)

    def test_hard_timeout_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(repo_lock_hard_timeout=-1)
        assert "repo_lock_hard_timeout" in str(exc_info.value)

    def test_hard_timeout_one_ok(self):
        s = Settings(repo_lock_hard_timeout=1)
        assert s.repo_lock_hard_timeout == 1

    def test_defaults_pass_validation(self):
        """Default values (300, 60, 3600) must pass validation."""
        s = Settings()
        assert s.repo_lock_idle_timeout == 300
        assert s.repo_lock_cleanup_interval == 60
        assert s.repo_lock_hard_timeout == 3600


# ---------------------------------------------------------------------------
# admin_user_ids — frozenset coercion (AC-R05)
# ---------------------------------------------------------------------------

class TestAdminUserIdsFrozenset:
    """Verify admin_user_ids is stored as frozenset for O(1) membership."""

    def test_admin_user_ids_is_frozenset(self):
        s = Settings(admin_user_ids=["ou_aaa", "ou_bbb"])
        assert isinstance(s.admin_user_ids, frozenset)
        assert "ou_aaa" in s.admin_user_ids
        assert "ou_bbb" in s.admin_user_ids
        assert len(s.admin_user_ids) == 2

    def test_admin_user_ids_empty_is_frozenset(self):
        s = Settings(_env_file=None)
        assert isinstance(s.admin_user_ids, frozenset)
        assert len(s.admin_user_ids) == 0

    def test_admin_user_ids_comma_string(self):
        s = Settings(admin_user_ids="ou_a,ou_b,ou_c")
        assert isinstance(s.admin_user_ids, frozenset)
        assert s.admin_user_ids == frozenset({"ou_a", "ou_b", "ou_c"})

    def test_admin_user_ids_membership_o1(self):
        """frozenset guarantees O(1) `in` checks."""
        s = Settings(admin_user_ids=["ou_x"])
        assert "ou_x" in s.admin_user_ids
        assert "ou_missing" not in s.admin_user_ids


# ---------------------------------------------------------------------------
# field_validator boundary tests — chat_lock_max_duration / cleanup_interval
# ---------------------------------------------------------------------------

class TestChatLockTimerValidation:
    """Verify that chat_lock_max_duration and chat_lock_cleanup_interval
    reject non-positive values (AC-R19)."""

    def test_chat_lock_max_duration_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(chat_lock_max_duration=0)
        assert "chat_lock_max_duration" in str(exc_info.value)

    def test_chat_lock_max_duration_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(chat_lock_max_duration=-1)
        assert "chat_lock_max_duration" in str(exc_info.value)

    def test_chat_lock_max_duration_one_ok(self):
        s = Settings(chat_lock_max_duration=1)
        assert s.chat_lock_max_duration == 1

    def test_chat_lock_cleanup_interval_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(chat_lock_cleanup_interval=0)
        assert "chat_lock_cleanup_interval" in str(exc_info.value)

    def test_chat_lock_cleanup_interval_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            Settings(chat_lock_cleanup_interval=-1)
        assert "chat_lock_cleanup_interval" in str(exc_info.value)

    def test_chat_lock_cleanup_interval_one_ok(self):
        s = Settings(chat_lock_cleanup_interval=1)
        assert s.chat_lock_cleanup_interval == 1

    def test_chat_lock_defaults_pass_validation(self):
        s = Settings()
        assert s.chat_lock_max_duration == 86400
        assert s.chat_lock_cleanup_interval == 60


# ---------------------------------------------------------------------------
# sandbox_strict_lock_mode default + sig_compat_deploy_date default
# ---------------------------------------------------------------------------

class TestNewConfigDefaults:
    """Verify new config items have correct defaults."""

    def test_sandbox_strict_lock_mode_default_false(self):
        s = Settings()
        assert s.sandbox_strict_lock_mode is False

    def test_sandbox_strict_lock_mode_true(self):
        s = Settings(sandbox_strict_lock_mode=True)
        assert s.sandbox_strict_lock_mode is True

    def test_sig_compat_deploy_date_default_empty(self):
        s = Settings()
        assert s.sig_compat_deploy_date == ""

    def test_spec_review_max_parallel_default(self):
        s = Settings()
        assert s.spec_review_max_parallel == 3

    def test_spec_max_cycles_default(self):
        s = Settings(_env_file=None)
        assert s.spec_max_cycles == 1000

    def test_slock_team_name_suffix_default_marks_slock_groups(self, monkeypatch):
        monkeypatch.delenv("SLOCK_TEAM_NAME_SUFFIX", raising=False)
        monkeypatch.delenv("SLOCK_TEAM_NAME_PREFIX", raising=False)
        s = Settings(_env_file=None)
        assert s.slock_team_name_suffix == "[Slock]"


# ---------------------------------------------------------------------------
# Settings env field defaults (merged from test_settings_env_fields.py)
# ---------------------------------------------------------------------------


class TestSettingsEnvFields:
    """Verify new fields have correct defaults and can be overridden via env."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SPEC_FAILED_TASK_ID_OVERRIDE", raising=False)
        monkeypatch.delenv("ACP_MAX_FILE_CHARS", raising=False)

        from src.config import Settings
        s = Settings()
        assert s.spec_failed_task_id_override == ""
        assert s.acp_max_file_chars == 200_000

    def test_env_override_spec_failed_task_id_override(self, monkeypatch):
        monkeypatch.setenv("SPEC_FAILED_TASK_ID_OVERRIDE", "some_task_123")
        from src.config import Settings
        s = Settings()
        assert s.spec_failed_task_id_override == "some_task_123"

    def test_env_override_acp_max_file_chars(self, monkeypatch):
        monkeypatch.setenv("ACP_MAX_FILE_CHARS", "500000")
        from src.config import Settings
        s = Settings()
        assert s.acp_max_file_chars == 500_000
