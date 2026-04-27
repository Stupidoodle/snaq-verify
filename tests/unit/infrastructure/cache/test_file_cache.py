"""Unit tests for FileCache."""

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from snaq_verify.infrastructure.cache.file_cache import FileCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache(tmp_path: Path) -> FileCache:
    """Return a fresh FileCache backed by a temporary directory."""
    return FileCache(tmp_path)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_set_and_get_returns_value(cache: FileCache) -> None:
    """Stored value is returned on get."""
    cache.set("fruit", "banana")
    assert cache.get("fruit") == "banana"


def test_set_complex_value(cache: FileCache) -> None:
    """Complex JSON-serializable values survive a write/read round-trip."""
    payload: dict[str, Any] = {
        "calories": 89,
        "items": [1, 2, 3],
        "nested": {"k": "v"},
    }
    cache.set("payload", payload)
    assert cache.get("payload") == payload


def test_get_missing_key_returns_none(cache: FileCache) -> None:
    """Missing key returns None without raising."""
    assert cache.get("does-not-exist") is None


def test_delete_removes_entry(cache: FileCache, tmp_path: Path) -> None:
    """Deleted key returns None on subsequent get."""
    cache.set("k", "v")
    cache.delete("k")
    assert cache.get("k") is None


def test_delete_missing_key_is_noop(cache: FileCache) -> None:
    """Deleting a non-existent key does not raise."""
    cache.delete("ghost")  # must not raise


def test_overwrite_replaces_value(cache: FileCache) -> None:
    """Setting the same key twice stores the latest value."""
    cache.set("x", 1)
    cache.set("x", 2)
    assert cache.get("x") == 2


def test_multiple_keys_are_independent(cache: FileCache) -> None:
    """Different keys use distinct files and don't interfere."""
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


# ---------------------------------------------------------------------------
# File layout tests
# ---------------------------------------------------------------------------


def test_one_file_per_key_is_created(cache: FileCache, tmp_path: Path) -> None:
    """Each stored key creates exactly one JSON file in cache_dir."""
    cache.set("alpha", "x")
    cache.set("beta", "y")
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 2


def test_file_contains_expected_fields(cache: FileCache, tmp_path: Path) -> None:
    """Cache file contains key, value, expires_at, and created_at fields."""
    cache.set("item", {"name": "chicken"}, ttl_seconds=3600)
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert data["key"] == "item"
    assert data["value"] == {"name": "chicken"}
    assert data["expires_at"] is not None
    assert data["created_at"] is not None


def test_no_ttl_results_in_null_expires_at(cache: FileCache, tmp_path: Path) -> None:
    """Entry without TTL has ``expires_at: null`` in the file."""
    cache.set("eternal", 42)
    json_files = list(tmp_path.glob("*.json"))
    data = json.loads(json_files[0].read_text())
    assert data["expires_at"] is None


def test_cache_dir_is_created_if_missing(tmp_path: Path) -> None:
    """FileCache creates the cache directory if it doesn't exist."""
    subdir = tmp_path / "deep" / "nested"
    assert not subdir.exists()
    fc = FileCache(subdir)
    fc.set("x", 1)
    assert subdir.exists()


# ---------------------------------------------------------------------------
# TTL / expiry tests
# ---------------------------------------------------------------------------


def test_entry_without_ttl_never_expires(cache: FileCache) -> None:
    """Entry stored without TTL is always returned."""
    cache.set("eternal", 42, ttl_seconds=None)
    assert cache.get("eternal") == 42


def test_entry_with_future_ttl_is_live(cache: FileCache) -> None:
    """Entry with a large TTL is returned before expiry."""
    cache.set("fresh", "data", ttl_seconds=3600)
    assert cache.get("fresh") == "data"


def test_expired_entry_returns_none(cache: FileCache, tmp_path: Path) -> None:
    """Expired entry returns None and is evicted from disk."""
    import hashlib

    key = "stale"
    digest = hashlib.sha256(key.encode()).hexdigest()
    path = tmp_path / f"{digest}.json"

    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    path.write_text(
        json.dumps(
            {
                "key": key,
                "value": "old-data",
                "expires_at": past,
                "created_at": past,
            }
        )
    )

    result = cache.get(key)
    assert result is None


def test_expired_entry_file_is_removed(cache: FileCache, tmp_path: Path) -> None:
    """Expired file is deleted from disk after get."""
    import hashlib

    key = "stale-file"
    digest = hashlib.sha256(key.encode()).hexdigest()
    path = tmp_path / f"{digest}.json"

    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    path.write_text(
        json.dumps(
            {
                "key": key,
                "value": "gone",
                "expires_at": past,
                "created_at": past,
            }
        )
    )

    cache.get(key)
    assert not path.exists()


# ---------------------------------------------------------------------------
# Corruption resilience
# ---------------------------------------------------------------------------


def test_corrupt_file_returns_none(cache: FileCache, tmp_path: Path) -> None:
    """A corrupt cache file is treated as a miss (no exception)."""
    import hashlib

    key = "corrupt"
    digest = hashlib.sha256(key.encode()).hexdigest()
    path = tmp_path / f"{digest}.json"
    path.write_text("this is not valid JSON!!!", encoding="utf-8")

    assert cache.get(key) is None


# ---------------------------------------------------------------------------
# Atomic write test
# ---------------------------------------------------------------------------


def test_atomic_write_no_tmp_file_left(cache: FileCache, tmp_path: Path) -> None:
    """After a successful set(), no .tmp file remains on disk."""
    cache.set("atomicity", {"value": 99})
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_concurrent_writes_are_safe(tmp_path: Path) -> None:
    """Concurrent writes from multiple threads don't corrupt the cache file.

    Two threads write different values for the same key repeatedly.  After all
    threads finish, the stored value must be one of the two valid payloads
    (not a mixture or a corrupt file).
    """
    cache = FileCache(tmp_path)
    key = "shared-key"
    errors: list[Exception] = []

    def writer(value: int, n: int) -> None:
        for _ in range(n):
            try:
                cache.set(key, value)
            except Exception as exc:
                errors.append(exc)

    t1 = threading.Thread(target=writer, args=(1, 50))
    t2 = threading.Thread(target=writer, args=(2, 50))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Exceptions during concurrent writes: {errors}"
    result = cache.get(key)
    assert result in (1, 2), f"Unexpected value: {result!r}"


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


def test_namespace_prefix_isolation(cache: FileCache) -> None:
    """Keys with different prefixes are stored independently."""
    cache.set("usda:search:chicken", [1, 2, 3])
    cache.set("tavily:search:chicken", ["snippet"])
    assert cache.get("usda:search:chicken") == [1, 2, 3]
    assert cache.get("tavily:search:chicken") == ["snippet"]
