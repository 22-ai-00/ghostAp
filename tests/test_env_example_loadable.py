"""End-to-end test: .env.example is loadable as valid Settings configuration.

Reads .env.example, strips all comment prefixes (#), injects required dummy
values, and validates that Settings can parse the result without exceptions.
"""

import re
from pathlib import Path

ENV_EXAMPLE_PATH = Path(__file__).parent.parent / ".env.example"

# Minimum required environment variables with valid dummy values.
# ADMIN_USER_IDS is a frozenset[str] — pydantic-settings tries JSON first,
# so provide a JSON array string for env var source compatibility.
REQUIRED_DUMMY_ENV = {
    "APP_ID": "cli_test_dummy_app_id",
    "APP_SECRET": "cli_test_dummy_app_secret",
    "ADMIN_USER_IDS": '["ou_test_admin_001"]',
}


class TestEnvExampleLoadable:
    """Verify .env.example can be parsed by Settings after uncommenting."""

    def test_env_example_file_exists(self):
        """Precondition: .env.example exists in project root."""
        assert ENV_EXAMPLE_PATH.exists(), f".env.example not found at {ENV_EXAMPLE_PATH}"

    def test_uncommented_env_is_valid_settings(self, monkeypatch, tmp_path):
        """Uncomment all lines in .env.example and validate via Settings.

        This catches type/format mismatches in newly added parameters that
        would otherwise only surface at runtime.
        """
        from src.config import Settings, _reset_settings_for_testing

        _reset_settings_for_testing()

        # Read and uncomment .env.example
        raw = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            # Skip decorative comment lines (boxes, headers, empty comments)
            if stripped.startswith("# ") and "=" not in stripped:
                continue
            if stripped == "#":
                continue
            # Uncomment lines that have KEY=VALUE after #
            if stripped.startswith("#") and "=" in stripped:
                # Remove leading # and optional space
                uncommented = re.sub(r"^#\s?", "", stripped)
                lines.append(uncommented)
            elif "=" in stripped and not stripped.startswith("#"):
                lines.append(stripped)

        # Filter out unparseable lines (descriptive comments that slipped through)
        # and fix empty complex-type fields that pydantic-settings can't JSON-parse
        valid_lines = []
        # Fields that are complex types (frozenset/list) and need JSON array format
        COMPLEX_FIELDS_DEFAULTS = {
            "ADMIN_USER_IDS": '["ou_test_admin"]',
        }
        for line in lines:
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # Skip lines where key contains spaces (not valid env var names)
            if " " in key:
                continue
            # Replace empty complex-type fields with valid JSON defaults
            if key in COMPLEX_FIELDS_DEFAULTS and not value.strip():
                valid_lines.append(f"{key}={COMPLEX_FIELDS_DEFAULTS[key]}")
            else:
                valid_lines.append(line)

        # Write to a temp .env file
        temp_env = tmp_path / ".env"
        temp_env.write_text("\n".join(valid_lines), encoding="utf-8")

        # Inject required dummy values via environment (overrides file values).
        # Environment variables take priority over .env file in pydantic-settings.
        for key, val in REQUIRED_DUMMY_ENV.items():
            monkeypatch.setenv(key, val)

        # Also set any values that would be empty strings but need content
        monkeypatch.setenv("DEFAULT_ACP_TOOL", "coco")

        # Validate Settings can load without exceptions
        settings = Settings(_env_file=str(temp_env))
        assert settings.app_id == "cli_test_dummy_app_id"
        assert settings.app_secret == "cli_test_dummy_app_secret"
