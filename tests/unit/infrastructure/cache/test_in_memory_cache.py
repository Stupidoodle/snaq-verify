"""Unit tests for InMemoryCache."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from snaq_verify.infrastructure.cache.in_memory_cache import InMemoryCache


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_set_and_get_returns_value() -> None:
    """Stored value is returned on get."""
    cache = InMemoryCache()
    cache.set("fruit", "banana")
    assert cache.get("fruit") == "banana"


def test_set_complex_value() -> None:
    """Complex JSON-serializable values are stored and returned intact."""
    cache = InMemoryCache()
    payload: dict[str, Any] = {"calories": 89, "items": [1, 2, 3], "nested": {"k": "v"}}
    cache.set("payload", payload)
    assert cache.get("payload") == payload


def test_get_missing_key_returns_none() -> None:
    """Missing key returns None without raising."""
    cache = InMemoryCache()
    assert cache.get("does-not-exist") is None


def test_delete_removes_entry() -> None:
    """Deleted key returns None on subsequent get."""
    cache = InMemoryCache()
    cache.set("k", "v")
    cache.delete("k")
    assert cache.get("k") is None


def test_delete_missing_key_is_noop() -> None:
    """Deleting a non-existent key does not raise."""
    cache = InMemoryCache()
    cache.delete("ghost")  # must not raise


def test_overwrite_replaces_value() -> None:
    """Setting the same key twice stores the latest value."""
    cache = InMemoryCache()
    cache.set("x", 1)
    cache.set("x", 2)
    assert cache.get("x") == 2


def test_multiple_keys_are_independent() -> None:
    """Different keys do not interfere with each other."""
    cache = InMemoryCache()
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_none_value_is_stored_and_returned() -> None:
    """Explicitly storing None is distinct from a missing key."""
    cache = InMemoryCache()
    cache.set("nullish", None)
    # get returns None both for missing and for stored-None, per the port contract.
    # The distinction here is that no KeyError is raised and len increases.
    assert len(cache) == 1


def test_clear_removes_all_entries() -> None:
    """clear() empties the cache completely."""
    cache = InMemoryCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


# ---------------------------------------------------------------------------
# TTL / expiry tests
# ---------------------------------------------------------------------------


def test_entry_without_ttl_never_expires() -> None:
    """Entry stored without TTL is always returned."""
    cache = InMemoryCache()
    cache.set("eternal", 42, ttl_seconds=None)
    assert cache.get("eternal") == 42


def test_entry_with_future_ttl_is_live() -> None:
    """Entry with a large TTL is returned before expiry."""
    cache = InMemoryCache()
    cache.set("fresh", "data", ttl_seconds=3600)
    assert cache.get("fresh") == "data"


def test_expired_entry_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired entry is evicted and returns None."""
    cache = InMemoryCache()
    # Store an entry that expired 1 second ago by manipulating the store directly.
    past = datetime.now(UTC) - timedelta(seconds=1)
    cache._store["stale"] = ("old-value", past)

    assert cache.get("stale") is None


def test_expired_entry_is_removed_from_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired entry is evicted (not merely hidden) on get."""
    cache = InMemoryCache()
    past = datetime.now(UTC) - timedelta(seconds=1)
    cache._store["stale"] = ("old-value", past)

    cache.get("stale")
    assert len(cache) == 0


def test_set_with_zero_ttl_expires_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TTL of 0 seconds makes the entry immediately expired.

    We monkeypatch datetime.now so the check fires *after* the set, simulating
    even one clock tick forward.
    """
    import snaq_verify.infrastructure.cache.in_memory_cache as mod

    original_now = datetime.now

    call_count = 0

    def patched_now(tz: Any = None) -> datetime:
        nonlocal call_count
        call_count += 1
        # First call (inside set) returns "now"; second call (inside get)
        # returns 1 second later so the entry looks expired.
        if call_count == 1:
            return original_now(UTC)
        return original_now(UTC) + timedelta(seconds=1)

    monkeypatch.setattr(mod, "datetime", type("dt", (), {"now": staticmethod(patched_now)})())

    cache = InMemoryCache()
    cache.set("instant-stale", "v", ttl_seconds=0)
    assert cache.get("instant-stale") is None


# ---------------------------------------------------------------------------
# Namespace isolation sanity check
# ---------------------------------------------------------------------------


def test_namespace_prefix_isolation() -> None:
    """Keys with different prefixes are independent — callers own namespacing."""
    cache = InMemoryCache()
    cache.set("usda:search:chicken", [1, 2, 3])
    cache.set("tavily:search:chicken", ["snippet"])
    assert cache.get("usda:search:chicken") == [1, 2, 3]
    assert cache.get("tavily:search:chicken") == ["snippet"]


# ---------------------------------------------------------------------------
# FakeCache truthiness
# ---------------------------------------------------------------------------


def test_fake_cache_empty_is_truthy() -> None:
    """FakeCache is truthy even when empty, so ``cache or FakeCache()`` is safe."""
    from tests.fakes.fake_cache import FakeCache

    cache = FakeCache()
    assert bool(cache) is True
    # Confirm the ``or`` idiom no longer discards the passed instance.
    result = cache or FakeCache()
    assert result is cache
