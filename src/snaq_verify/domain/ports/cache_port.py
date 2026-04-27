"""Cache port — key-value cache for source lookup responses."""

from abc import ABC, abstractmethod
from typing import Any


class CachePort(ABC):
    """Abstract interface for a JSON-serializable key-value cache.

    Adapters: `InMemoryCache` (default in tests), `FileCache` (default in
    runs). Drop-in replacement for distributed caches (Valkey/Redis) is
    intentional — only `bootstrap.py` would change.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached value for `key`, or None if absent or expired.

        Args:
            key: Cache key (caller is responsible for namespacing).

        Returns:
            The previously stored value (after JSON round-trip), or None.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store `value` under `key`, optionally with a TTL.

        Args:
            key: Cache key.
            value: Any JSON-serializable payload.
            ttl_seconds: Time-to-live in seconds. None means no expiry.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Drop the entry for `key`. No-op if missing."""
        raise NotImplementedError
