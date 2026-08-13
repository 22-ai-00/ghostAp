"""Normalize third-party Lark SDK logging for the GhostAP service."""

from __future__ import annotations

import logging

_LARK_LOGGER_NAME = "Lark"
_RECOVERABLE_ABNORMAL_CLOSE = (
    "receive message loop exit, err: no close frame received or sent"
)


class _RecoverableDisconnectFilter(logging.Filter):
    """Treat an auto-reconnecting abnormal close as a warning, not a crash."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.levelno >= logging.ERROR
            and _RECOVERABLE_ABNORMAL_CLOSE in record.getMessage()
        ):
            record.levelno = logging.WARNING
            record.levelname = logging.getLevelName(logging.WARNING)
        return True


def configure_lark_sdk_logging() -> None:
    """Route the shared SDK logger through the application exactly once."""

    sdk_logger = logging.getLogger(_LARK_LOGGER_NAME)
    for handler in tuple(sdk_logger.handlers):
        sdk_logger.removeHandler(handler)
    sdk_logger.propagate = True
    if not any(
        isinstance(log_filter, _RecoverableDisconnectFilter)
        for log_filter in sdk_logger.filters
    ):
        sdk_logger.addFilter(_RecoverableDisconnectFilter())


__all__ = ["configure_lark_sdk_logging"]
