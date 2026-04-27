"""FakeCache — InMemoryCache subclass for test isolation."""

from snaq_verify.infrastructure.cache.in_memory_cache import InMemoryCache


class FakeCache(InMemoryCache):
    """Drop-in CachePort fake for unit tests.

    Extends :class:`InMemoryCache` with no additional behaviour; all cache
    operations are identical.  Using a distinct class makes test intent
    explicit (\"this test uses a fake cache\") and keeps the import story
    clean for teammates who inject ``FakeCache`` in their test modules.

    **Truthiness:** always ``True``, even when empty.  This prevents the
    common ``cache or FakeCache()`` helper pattern from silently discarding
    a passed-in instance because an empty cache has ``len() == 0``.

    Example::

        cache = FakeCache()
        client = TavilyClient(api_key=\"...\", cache=cache, logger=FakeLogger())

        # after exercising the client:
        assert cache.get(\"tavily:chicken:5:v1\") is not None

        # reset between test cases:
        cache.clear()
    """

    def __bool__(self) -> bool:
        """Always truthy — an empty cache is still a valid cache instance."""
        return True
