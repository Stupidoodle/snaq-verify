"""FakeLogger — in-memory LoggerPort implementation for tests."""

from typing import Any

from snaq_verify.domain.ports.logger_port import LoggerPort


class FakeLogger(LoggerPort):
    """Captures structured log calls for assertion in unit tests.

    Each call is appended to ``messages`` as a 3-tuple::

        (level: str, message: str, context: dict[str, Any])

    Example::

        logger = FakeLogger()
        runner.run(logger=logger)

        assert ("info", "pipeline started", {}) in logger.messages
        assert logger.at_level("error") == []
    """

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # LoggerPort implementation
    # ------------------------------------------------------------------

    def debug(self, message: str, **kwargs: Any) -> None:
        self.messages.append(("debug", message, kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self.messages.append(("info", message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self.messages.append(("warning", message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self.messages.append(("error", message, kwargs))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def at_level(self, level: str) -> list[tuple[str, str, dict[str, Any]]]:
        """Return only the entries recorded at *level*."""
        return [entry for entry in self.messages if entry[0] == level]

    def reset(self) -> None:
        """Clear all captured messages (useful between test cases)."""
        self.messages.clear()
