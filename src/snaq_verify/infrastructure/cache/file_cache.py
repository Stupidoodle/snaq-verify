"""File-based cache adapter — persists JSON entries under a configurable directory."""

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from snaq_verify.domain.ports.cache_port import CachePort

_ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"


class FileCache(CachePort):
    """Disk-backed key-value cache that persists between process restarts.

    Layout::

        {cache_dir}/
            <sha256(key)>.json     # one file per cache entry
            <sha256(key)>.json.tmp # staging area for atomic writes

    Each file stores a JSON object with four fields:

    .. code-block:: json

        {
            "key":        "<original key string>",
            "value":      <any JSON-serializable payload>,
            "expires_at": "<ISO-8601 UTC string> | null",
            "created_at": "<ISO-8601 UTC string>"
        }

    Writes are atomic: the payload is written to a ``<name>.tmp`` file first,
    then renamed into place via ``os.replace`` (POSIX rename-on-same-filesystem
    is guaranteed atomic).

    Expired entries are evicted lazily on the first ``get`` that observes them.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialise the cache.

        Args:
            cache_dir: Directory under which cache files are stored.  Created
                automatically if it does not exist.
        """
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CachePort interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` if absent or expired.

        Expired entries are removed from disk before returning ``None``.

        Args:
            key: Cache key (caller is responsible for namespacing).

        Returns:
            The stored value (after a JSON round-trip), or ``None``.
        """
        path = self._path_for(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt or inaccessible file — treat as cache miss.
            return None

        expires_at_raw: str | None = data.get("expires_at")
        if expires_at_raw is not None:
            expires_at = datetime.fromisoformat(expires_at_raw)
            if datetime.now(UTC) >= expires_at:
                # Evict the stale file.
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                return None

        return data.get("value")

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store *value* under *key*, optionally expiring after *ttl_seconds*.

        Uses an atomic write (tmp → rename) to avoid partial-read races.

        Args:
            key: Cache key.
            value: Any JSON-serializable payload.
            ttl_seconds: Seconds until expiry.  ``None`` means no expiry.
        """
        now = datetime.now(UTC)
        expires_at: str | None = None
        if ttl_seconds is not None:
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()

        payload = {
            "key": key,
            "value": value,
            "expires_at": expires_at,
            "created_at": now.isoformat(),
        }
        serialized = json.dumps(payload, ensure_ascii=False)

        target = self._path_for(key)
        # Use a unique suffix per write so concurrent writes for the same key
        # don't overwrite each other's staging file before os.replace() fires.
        tmp = target.with_name(f"{target.stem}.{secrets.token_hex(8)}.tmp")

        tmp.write_text(serialized, encoding="utf-8")
        os.replace(tmp, target)

    def delete(self, key: str) -> None:
        """Drop the entry for *key*.  No-op if missing.

        Args:
            key: Cache key.
        """
        path = self._path_for(key)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        """Return the on-disk path for *key*.

        The key is SHA-256 hashed to produce a filesystem-safe filename.

        Args:
            key: Cache key.

        Returns:
            Absolute ``Path`` to the JSON file (may not exist).
        """
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{digest}.json"
