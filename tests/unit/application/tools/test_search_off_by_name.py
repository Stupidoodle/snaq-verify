"""Unit tests for the search_off_by_name tool factory."""

import pytest
from agents import FunctionTool

from snaq_verify.application.tools.search_off_by_name import make_search_off_by_name
from snaq_verify.domain.models.source_lookup import OFFProduct
from tests.fakes.fake_open_food_facts_client import FakeOpenFoodFactsClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAGE_PRODUCT = OFFProduct(
    code="3564700002449",
    product_name="Total 0% Greek Yogurt",
    brands="Fage",
)
_OTHER_PRODUCT = OFFProduct(
    code="1234567890001",
    product_name="Total 2% Greek Yogurt",
    brands="Fage",
)

_SEARCH_NAME = "Total 0% Greek Yogurt"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_products() -> None:
    """Raw function returns matching OFFProduct list when results exist."""
    client = FakeOpenFoodFactsClient(
        search_map={_SEARCH_NAME: [_FAGE_PRODUCT, _OTHER_PRODUCT]}
    )
    fn, _ = make_search_off_by_name(client)

    results = await fn(_SEARCH_NAME)

    assert len(results) == 2
    assert results[0].product_name == "Total 0% Greek Yogurt"
    assert results[1].product_name == "Total 2% Greek Yogurt"


@pytest.mark.asyncio
async def test_returns_empty_when_no_results() -> None:
    """Unknown name → empty list, not an exception."""
    client = FakeOpenFoodFactsClient()
    fn, _ = make_search_off_by_name(client)

    results = await fn("xyzzy_nothing_matches")

    assert results == []


@pytest.mark.asyncio
async def test_forwards_brand_filter() -> None:
    """Brand argument is passed through to the client."""
    client = FakeOpenFoodFactsClient(
        search_map={_SEARCH_NAME: [_FAGE_PRODUCT]}
    )
    fn, _ = make_search_off_by_name(client)

    await fn(_SEARCH_NAME, brand="Fage")

    assert client.search_calls == [(_SEARCH_NAME, "Fage", 10)]


@pytest.mark.asyncio
async def test_forwards_page_size() -> None:
    """page_size is forwarded to the client and respected."""
    many = [
        OFFProduct(code=str(i), product_name=f"Product {i}")
        for i in range(8)
    ]
    client = FakeOpenFoodFactsClient(search_map={_SEARCH_NAME: many})
    fn, _ = make_search_off_by_name(client)

    results = await fn(_SEARCH_NAME, page_size=3)

    # FakeOpenFoodFactsClient truncates to page_size
    assert len(results) == 3
    assert client.search_calls == [(_SEARCH_NAME, None, 3)]


@pytest.mark.asyncio
async def test_tracks_search_call() -> None:
    """The fake records (name, brand, page_size) for every call."""
    client = FakeOpenFoodFactsClient(
        search_map={_SEARCH_NAME: [_FAGE_PRODUCT]}
    )
    fn, _ = make_search_off_by_name(client)

    await fn(_SEARCH_NAME, brand="Fage", page_size=5)

    assert client.search_calls == [(_SEARCH_NAME, "Fage", 5)]


@pytest.mark.asyncio
async def test_default_page_size_is_ten() -> None:
    """Omitting page_size uses the default of 10."""
    client = FakeOpenFoodFactsClient()
    fn, _ = make_search_off_by_name(client)

    await fn(_SEARCH_NAME)

    _, _, recorded_page_size = client.search_calls[0]
    assert recorded_page_size == 10


def test_tool_export_is_function_tool() -> None:
    """Factory exports a FunctionTool suitable for Agent(tools=[...])."""
    client = FakeOpenFoodFactsClient()
    _, tool = make_search_off_by_name(client)

    assert isinstance(tool, FunctionTool)
    assert tool.name == "search_off_by_name"
