"""Unit tests for OpenFoodFactsClient."""

import pytest
import respx
from httpx import Response

from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.infrastructure.sources.open_food_facts_client import (
    OpenFoodFactsClient,
    _NOT_FOUND_SENTINEL,
)
from tests.fakes.fake_cache import FakeCache
from tests.fakes.fake_logger import FakeLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OFF_BASE_URL = "https://world.openfoodfacts.org"
BARCODE_URL = f"{OFF_BASE_URL}/api/v2/product/{{barcode}}.json"
SEARCH_URL = f"{OFF_BASE_URL}/cgi/search.pl"

#: The Fage Greek Yogurt barcode that OFF does not have — must return None.
FAGE_BARCODE = "5200435000027"

#: A well-known barcode (Nutella) with complete nutriment data.
NUTELLA_BARCODE = "3017620422003"

_NUTELLA_NUTRIMENTS = {
    "energy-kcal_100g": 539,
    "proteins_100g": 6.3,
    "fat_100g": 30.9,
    "saturated-fat_100g": 10.6,
    "carbohydrates_100g": 57.5,
    "sugars_100g": 56.3,
    "fiber_100g": 0.0,
    "sodium_100g": 0.107,  # grams → adapter multiplies by 1000 → 107 mg
}

_NUTELLA_PRODUCT_RAW = {
    "code": NUTELLA_BARCODE,
    "product_name": "Nutella",
    "brands": "Ferrero",
    "completeness": 0.875,
    "popularity_key": 28000,
    "nutriments": _NUTELLA_NUTRIMENTS,
}

_NUTELLA_RESPONSE = {"status": 1, "product": _NUTELLA_PRODUCT_RAW}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(cache: FakeCache | None = None) -> OpenFoodFactsClient:
    from snaq_verify.core.config import Settings

    settings = Settings(
        USDA_API_KEY="x",
        OPENAI_API_KEY="x",
        TAVILY_API_KEY="x",
        OFF_BASE_URL=OFF_BASE_URL,
        OFF_USER_AGENT="snaq-verify-test/0.1",
        HTTP_TIMEOUT_SECONDS=10.0,
        CACHE_TTL_DAYS=30,
    )
    # NOTE: use `is not None` — FakeCache() is falsy when empty (len == 0),
    # so `cache or FakeCache()` would silently discard the passed instance.
    return OpenFoodFactsClient(
        settings=settings,
        logger=FakeLogger(),
        cache=cache if cache is not None else FakeCache(),
    )


# ---------------------------------------------------------------------------
# lookup_by_barcode — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_returns_product() -> None:
    """200 with full nutriments → OFFProduct with nutrition populated."""
    respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        return_value=Response(200, json=_NUTELLA_RESPONSE)
    )
    client = _make_client()
    product = await client.lookup_by_barcode(NUTELLA_BARCODE)

    assert product is not None
    assert isinstance(product, OFFProduct)
    assert product.code == NUTELLA_BARCODE
    assert product.product_name == "Nutella"
    assert product.brands == "Ferrero"
    assert product.nutrition_per_100g is not None
    assert product.nutrition_per_100g.calories_kcal == pytest.approx(539.0)
    assert product.nutrition_per_100g.sodium_mg == pytest.approx(107.0)


# ---------------------------------------------------------------------------
# lookup_by_barcode — Fage 404 case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_fage_404() -> None:
    """OFF 404 for Fage barcode → returns None (not an exception)."""
    respx.get(BARCODE_URL.format(barcode=FAGE_BARCODE)).mock(
        return_value=Response(404)
    )
    client = _make_client()
    result = await client.lookup_by_barcode(FAGE_BARCODE)
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_status_zero_returns_none() -> None:
    """OFF returns 200 with status=0 for invalid barcodes → None."""
    respx.get(BARCODE_URL.format(barcode="0000000000000")).mock(
        return_value=Response(200, json={"status": 0, "status_verbose": "product not found"})
    )
    client = _make_client()
    result = await client.lookup_by_barcode("0000000000000")
    assert result is None


# ---------------------------------------------------------------------------
# lookup_by_barcode — missing / incomplete nutriments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_missing_nutriments() -> None:
    """Product with no nutriments dict → OFFProduct with nutrition_per_100g=None."""
    raw = {**_NUTELLA_PRODUCT_RAW}
    raw.pop("nutriments")
    respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        return_value=Response(200, json={"status": 1, "product": raw})
    )
    client = _make_client()
    product = await client.lookup_by_barcode(NUTELLA_BARCODE)
    assert product is not None
    assert product.nutrition_per_100g is None


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_partial_nutriments() -> None:
    """Nutriments missing a required field → nutrition_per_100g=None."""
    partial = {k: v for k, v in _NUTELLA_NUTRIMENTS.items() if k != "fiber_100g"}
    raw = {**_NUTELLA_PRODUCT_RAW, "nutriments": partial}
    respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        return_value=Response(200, json={"status": 1, "product": raw})
    )
    client = _make_client()
    product = await client.lookup_by_barcode(NUTELLA_BARCODE)
    assert product is not None
    assert product.nutrition_per_100g is None


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_salt_fallback() -> None:
    """When sodium_100g absent but salt_100g present, sodium is derived."""
    # salt_100g=0.2675 g → sodium_mg = 0.2675 * 1000 / 2.5 = 107 mg
    nutriments = {**_NUTELLA_NUTRIMENTS}
    del nutriments["sodium_100g"]
    nutriments["salt_100g"] = 0.2675
    raw = {**_NUTELLA_PRODUCT_RAW, "nutriments": nutriments}
    respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        return_value=Response(200, json={"status": 1, "product": raw})
    )
    client = _make_client()
    product = await client.lookup_by_barcode(NUTELLA_BARCODE)
    assert product is not None
    assert product.nutrition_per_100g is not None
    assert product.nutrition_per_100g.sodium_mg == pytest.approx(107.0, rel=1e-3)


# ---------------------------------------------------------------------------
# lookup_by_barcode — cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_caches_result() -> None:
    """After a successful fetch, the result is stored in cache."""
    respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        return_value=Response(200, json=_NUTELLA_RESPONSE)
    )
    cache = FakeCache()
    client = _make_client(cache=cache)
    await client.lookup_by_barcode(NUTELLA_BARCODE)

    cached = cache.get(f"off:barcode:{NUTELLA_BARCODE}:v1")
    assert cached is not None
    assert isinstance(cached, dict)
    assert cached["code"] == NUTELLA_BARCODE


@pytest.mark.asyncio
async def test_lookup_by_barcode_cache_hit_skips_api() -> None:
    """Pre-seeded cache → no HTTP call made."""
    cache = FakeCache()
    product = OFFProduct(code=NUTELLA_BARCODE, product_name="Nutella")
    cache.set(f"off:barcode:{NUTELLA_BARCODE}:v1", product.model_dump())

    # No respx mock — any real HTTP call would raise ConnectionError.
    client = _make_client(cache=cache)
    result = await client.lookup_by_barcode(NUTELLA_BARCODE)
    assert result is not None
    assert result.code == NUTELLA_BARCODE
    assert result.product_name == "Nutella"


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_cache_miss_then_hit() -> None:
    """Second identical lookup uses cache — API called exactly once."""
    route = respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        return_value=Response(200, json=_NUTELLA_RESPONSE)
    )
    cache = FakeCache()
    client = _make_client(cache=cache)

    await client.lookup_by_barcode(NUTELLA_BARCODE)
    await client.lookup_by_barcode(NUTELLA_BARCODE)

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_caches_404_sentinel() -> None:
    """404 response is cached as sentinel so the network isn't hit again."""
    route = respx.get(BARCODE_URL.format(barcode=FAGE_BARCODE)).mock(
        return_value=Response(404)
    )
    cache = FakeCache()
    client = _make_client(cache=cache)

    await client.lookup_by_barcode(FAGE_BARCODE)
    result = await client.lookup_by_barcode(FAGE_BARCODE)

    assert result is None
    assert route.call_count == 1
    assert cache.get(f"off:barcode:{FAGE_BARCODE}:v1") == _NOT_FOUND_SENTINEL


# ---------------------------------------------------------------------------
# lookup_by_barcode — error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_lookup_by_barcode_network_error_raises() -> None:
    """Network errors propagate — not swallowed."""
    import httpx

    respx.get(BARCODE_URL.format(barcode=NUTELLA_BARCODE)).mock(
        side_effect=httpx.ConnectError("timeout")
    )
    client = _make_client()
    with pytest.raises(Exception):
        await client.lookup_by_barcode(NUTELLA_BARCODE)


# ---------------------------------------------------------------------------
# search_by_name — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_search_returns_products() -> None:
    """200 with products list → list[OFFProduct]."""
    respx.get(SEARCH_URL).mock(
        return_value=Response(
            200,
            json={"count": 1, "products": [_NUTELLA_PRODUCT_RAW]},
        )
    )
    client = _make_client()
    results = await client.search_by_name("Nutella")

    assert len(results) == 1
    assert results[0].product_name == "Nutella"
    assert results[0].brands == "Ferrero"
    assert results[0].nutrition_per_100g is not None


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_results() -> None:
    """Empty products array → empty list."""
    respx.get(SEARCH_URL).mock(
        return_value=Response(200, json={"count": 0, "products": []})
    )
    client = _make_client()
    results = await client.search_by_name("xyzzy_no_match_abc")
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_search_with_brand_sends_param() -> None:
    """Brand filter is forwarded as query param."""
    route = respx.get(SEARCH_URL).mock(
        return_value=Response(200, json={"count": 0, "products": []})
    )
    client = _make_client()
    await client.search_by_name("Total 0% Greek Yogurt", brand="Fage")

    assert route.called
    request = route.calls[0].request
    assert b"Fage" in request.url.query or "Fage" in str(request.url)


# ---------------------------------------------------------------------------
# search_by_name — cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_search_caches_result() -> None:
    """After a successful search, results are stored in cache."""
    respx.get(SEARCH_URL).mock(
        return_value=Response(
            200, json={"count": 1, "products": [_NUTELLA_PRODUCT_RAW]}
        )
    )
    cache = FakeCache()
    client = _make_client(cache=cache)
    await client.search_by_name("Nutella", page_size=5)

    cache_key = "off:search:nutella::5:v1"
    cached = cache.get(cache_key)
    assert cached is not None
    assert isinstance(cached, list)
    assert len(cached) == 1


@pytest.mark.asyncio
async def test_search_cache_hit_skips_api() -> None:
    """Pre-seeded cache → no HTTP call made."""
    cache = FakeCache()
    product = OFFProduct(code=NUTELLA_BARCODE, product_name="Nutella")
    cache.set("off:search:nutella::10:v1", [product.model_dump()])

    # No respx mock — any real HTTP call would raise.
    client = _make_client(cache=cache)
    results = await client.search_by_name("Nutella")
    assert len(results) == 1
    assert results[0].product_name == "Nutella"


@pytest.mark.asyncio
@respx.mock
async def test_search_cache_miss_then_hit() -> None:
    """Second identical search uses cache — API called exactly once."""
    route = respx.get(SEARCH_URL).mock(
        return_value=Response(
            200, json={"count": 1, "products": [_NUTELLA_PRODUCT_RAW]}
        )
    )
    cache = FakeCache()
    client = _make_client(cache=cache)

    await client.search_by_name("Nutella")
    await client.search_by_name("Nutella")

    assert route.call_count == 1
