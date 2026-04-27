"""Structlog-backed logger adapter."""

import logging
from typing import Any

import structlog

from snaq_verify.domain.ports.logger_port import LoggerPort

# Guards against re-configuring structlog after ``cache_logger_on_first_use``
# has frozen the first-use configuration.  Once True, further calls to
# ``_configure_once`` are no-ops.
_configured: bool = False


def _configure_once(log_level: str = "INFO") -> None:
    """Configure structlog idempotently.

    Sets up an always-JSON pipeline so output is machine-parseable in CI and
    in production.  Safe to call multiple times; only the first call has any
    effect (``cache_logger_on_first_use=True`` seals the configuration).

    Args:
        log_level: Standard library log level string (e.g. ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``).
    """
    global _configured
    if _configured:
        return

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    _configured = True


class StructlogLogger(LoggerPort):
    """Structured JSON logger backed by ``structlog>=25.5``.

    Configures structlog once per process on first instantiation.  Subsequent
    instances share the same global configuration (structlog is process-global
    by design).

    All ``**kwargs`` passed to each log method are forwarded as structured
    fields in the JSON output, e.g.::

        logger.info("cache hit", key="usda:search:chicken:v1", ttl=86400)
        # → {"event": "cache hit", "key": "usda:search:chicken:v1",
        #    "ttl": 86400, "level": "info", "timestamp": "..."}

    Args:
        log_level: Standard library log level string (case-insensitive).
            Defaults to ``"INFO"``.
    """

    def __init__(self, log_level: str = "INFO") -> None:
        _configure_once(log_level)
        self._log: structlog.stdlib.BoundLogger = structlog.get_logger("snaq_verify")

    # ------------------------------------------------------------------
    # LoggerPort implementation
    # ------------------------------------------------------------------

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug-level message with structured context.

        Args:
            message: Human-readable event description.
            **kwargs: Arbitrary structured context fields.
        """
        self._log.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info-level message with structured context.

        Args:
            message: Human-readable event description.
            **kwargs: Arbitrary structured context fields.
        """
        self._log.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning-level message with structured context.

        Args:
            message: Human-readable event description.
            **kwargs: Arbitrary structured context fields.
        """
        self._log.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error-level message with structured context.

        Args:
            message: Human-readable event description.
            **kwargs: Arbitrary structured context fields.
        """
        self._log.error(message, **kwargs)
