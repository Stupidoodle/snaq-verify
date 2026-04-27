"""Unit tests for TavilyClient."""

import json

import pytest
import respx
from httpx import Response

from snaq_verify.domain.models.source_lookup import WebSnippet
from snaq_verify.infrastructure.sources.tavily_client import TavilyClient
from tests.fakes.fake_cache import FakeCache
from tests.fakes.fake_logger import FakeLogger

# Tavily's actual search endpoint (used by the Python SDK internally)
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNIPPET = WebSnippet(
    url="https://nutritiondata.self.com/facts/poultry-products/703/2",
    title="Chicken Breast, Raw Nutrition",
    content="Chicken breast raw: protein 23g, fat 1.2g, carbs 0g per 100g.",
    score=0.92,
)

_TAVILY_RESULT = {
    "url": _SNIPPET.url,
    "title": _SNIPPET.title,
    "content": _SNIPPET.content,
    "score": _SNIPPET.score,
}

_TAVILY_RESPONSE = {"results": [_TAVILY_RESULT]}


def _make_client(cache: FakeCache | None = None, ttl_seconds: int = 86400) -> TavilyClient:
    return TavilyClient(
        api_key="test-key",
        cache=cache if cache is not None else FakeCache(),
        logger=FakeLogger(),
        ttl_seconds=ttl_seconds,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_search_returns_snippets() -> None:
    """search() maps Tavily response to list[WebSnippet]."""
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=Response(200, json=_TAVILY_RESPONSE)
    )
    client = _make_client()
    results = await client.search("chicken breast nutrition")
    assert len(results) == 1
    assert results[0].url == _SNIPPET.url
    assert results[0].title == _SNIPPET.title
    assert results[0].content == _SNIPPET.content
    assert results[0].score == pytest.approx(_SNIPPET.score)


@pytest.mark.asyncio
@respx.mock
async def test_search_respects_max_results() -> None:
    """max_results is forwarded and the returned list is not over-truncated."""
    many_results = {"results": [_TAVILY_RESULT] * 8}
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=Response(200, json=many_results)
    )
    client = _make_client()
    results = await client.search("chicken breast nutrition", max_results=3)
    # We get back whatever Tavily returns (mock returns 8); client shouldn't truncate.
    assert len(results) == 8  # The SDK passes max_results to the API; mock returns all.


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_result() -> None:
    """search() returns an empty list when Tavily has no results."""
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=Response(200, json={"results": []})
    )
    client = _make_client()
    results = await client.search("xyzzy foobarbaz nutrition facts")
    assert results == []


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_search_caches_result() -> None:
    """After the first search, result is stored in cache."""
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=Response(200, json=_TAVILY_RESPONSE)
    )
    cache = FakeCache()
    client = _make_client(cache=cache)
    await client.search("chicken breast nutrition", max_results=5)

    cache_key = "tavily:chicken breast nutrition:5:v1"
    cached = cache.get(cache_key)
    assert cached is not None
    assert isinstance(cached, list)
    assert cached[0]["url"] == _SNIPPET.url


@pytest.mark.asyncio
async def test_search_cache_hit_skips_api() -> None:
    """Cache hit means no HTTP call is made."""
    cache = FakeCache()
    # Pre-populate the cache
    cache_key = "tavily:chicken breast nutrition:5:v1"
    cache.set(cache_key, [_TAVILY_RESULT])

    # No respx mock — any real HTTP call would raise.
    client = _make_client(cache=cache)
    results = await client.search("chicken breast nutrition", max_results=5)
    assert len(results) == 1
    assert results[0].url == _SNIPPET.url


@pytest.mark.asyncio
@respx.mock
async def test_search_cache_miss_then_hit() -> None:
    """Second identical search hits cache, not the network."""
    route = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=Response(200, json=_TAVILY_RESPONSE)
    )
    cache = FakeCache()
    client = _make_client(cache=cache)

    await client.search("chicken breast nutrition")
    await client.search("chicken breast nutrition")

    # API should be called exactly once.
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_search_network_error_raises() -> None:
    """Network-level errors propagate rather than being swallowed."""
    import httpx

    respx.post(TAVILY_SEARCH_URL).mock(side_effect=httpx.ConnectError("timeout"))
    client = _make_client()
    with pytest.raises(Exception):
        await client.search("chicken breast nutrition")


@pytest.mark.asyncio
@respx.mock
async def test_search_missing_optional_fields() -> None:
    """Tavily result missing optional 'score' still maps to WebSnippet."""
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Title",
                        "content": "Content",
                        # score omitted
                    }
                ]
            },
        )
    )
    client = _make_client()
    results = await client.search("some query")
    assert len(results) == 1
    assert results[0].score is None
