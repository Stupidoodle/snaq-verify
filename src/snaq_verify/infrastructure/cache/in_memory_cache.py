"""In-memory cache adapter — drop-in fake for tests and local dev."""

from datetime import UTC, datetime
from typing import Any

from snaq_verify.domain.ports.cache_port import CachePort


class InMemoryCache(CachePort):
    """Thread-unsafe in-memory key-value cache with optional TTL.

    Intended for unit tests and local experimentation where a real file-system
    cache would add unnecessary I/O.  Drop-in replacement for `FileCache`.

    The TTL is stored as an absolute UTC expiry timestamp.  Expired entries are
    evicted lazily on the first `get` that observes them.
    """

    def __init__(self) -> None:
        # Mapping of key -> (value, expires_at or None)
        self._store: dict[str, tuple[Any, datetime | None]] = {}

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` if absent or expired.

        Args:
            key: Cache key (caller is responsible for namespacing).

        Returns:
            The stored value, or ``None`` if the entry is missing or expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        value, expires_at = entry
        if expires_at is not None and datetime.now(UTC) >= expires_at:
            del self._store[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store *value* under *key*, optionally expiring after *ttl_seconds*.

        Args:
            key: Cache key.
            value: Any JSON-serializable payload.
            ttl_seconds: Seconds until expiry.  ``None`` means no expiry.
        """
        expires_at: datetime | None = None
        if ttl_seconds is not None:
            from datetime import timedelta

            expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """Drop the entry for *key*.  No-op if missing.

        Args:
            key: Cache key.
        """
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries.  Useful for test teardown."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of entries (including potentially expired ones)."""
        return len(self._store)
