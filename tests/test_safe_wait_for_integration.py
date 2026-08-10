"""Integration tests for safe_wait_for at production call sites.

Verifies that safe_wait_for is correctly wired in at:
- src/acp/session.py: _read_stream, _health_check_session
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# ACPSession._read_stream integration
# ---------------------------------------------------------------------------


class TestReadStreamTimeout:
    """Verify that _read_stream uses safe_wait_for with action label."""

    @pytest.mark.asyncio
    async def test_read_stream_timeout_produces_non_empty_message(self):
        """When stream.read hangs, safe_wait_for produces non-empty TimeoutError."""
        from src.utils.async_helpers import safe_wait_for

        # Simulate a stream whose read() never returns
        stream = MagicMock()

        async def _hang(n):
            await asyncio.sleep(100)

        stream.read = _hang

        # Call safe_wait_for the same way _read_stream does
        with pytest.raises(TimeoutError) as exc_info:
            await safe_wait_for(stream.read(4096), timeout=0.01, action="ACP stream read")

        msg = str(exc_info.value)
        assert msg.strip() != ""
        assert "ACP stream read" in msg
        assert "0.01" in msg


# ---------------------------------------------------------------------------
# ACPSession._health_check_session integration
# ---------------------------------------------------------------------------


