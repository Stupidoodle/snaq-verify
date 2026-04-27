"""Logger port — structured-logging interface."""

from abc import ABC, abstractmethod
from typing import Any


class LoggerPort(ABC):
    """Abstract interface for structured logging.

    Adapters should accept arbitrary keyword arguments as structured context
    and not perform formatting on the message itself.
    """

    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug-level message with structured context."""
        raise NotImplementedError

    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info-level message with structured context."""
        raise NotImplementedError

    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning-level message with structured context."""
        raise NotImplementedError

    @abstractmethod
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error-level message with structured context."""
        raise NotImplementedError
