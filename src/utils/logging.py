"""Structured logging utilities.

Provides a JSON formatter and a logger factory for consistent log output.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class StructuredFormatter(logging.Formatter):
    """Format log records as JSON for machine-friendly ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a single log record as a JSON string."""
        payload: dict[str, str] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured with the structured JSON formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    has_structured_handler = any(
        isinstance(handler.formatter, StructuredFormatter) for handler in logger.handlers
    )
    if not has_structured_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger
