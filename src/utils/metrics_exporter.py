"""Logging-backed review metrics exporter."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class LoggerExporter:
    def export_metrics(self, metrics: dict, *, prefix: str = "") -> None:
        try:
            logger.info("%s review_metrics: %s", prefix, json.dumps(metrics, ensure_ascii=False))
        except Exception:
            logger.debug("failed to export metrics via logger", exc_info=True)


_exporter_instance = LoggerExporter()


def get_metrics_exporter() -> LoggerExporter:
    return _exporter_instance
