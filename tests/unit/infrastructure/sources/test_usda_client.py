"""Tests for USDAClient — uses respx to mock httpx calls."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.domain.models.source_lookup import USDACandidate
from snaq_verify.infrastructure.cache.in_memory_cache import InMemoryCache
from snaq_verify.infrastructure.sources.usda_client import USDAClient, _is_likely_valid
from tests.fakes.fake_logger import FakeLogger

# ---------------------------------------------------------------------------
# Helpers — load pinned fixture files
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parents[3] / "data" / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


SEARCH_RESPONSE = _load("usda_search_chicken_breast_synthetic.json")
FOOD_RESPONSE = _load("usda_food_chicken_breast_raw_synthetic.json")
EMPTY_SEARCH = _load("usda_search_empty.json")
MISSING_NUTRIENTS = _load("usda_food_missing_nutrients.json")

BASE_URL = "https://api.nal.usda.gov/fdc/v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        USDA_API_KEY="test-key",
        OPENAI_API_KEY="openai-fake",
        TAVILY_API_KEY="tavily-fake",
    )


@pytest.fixture
def cache() -> InMemoryCache:
    return InMemoryCache()


@pytest.fixture
def logger() -> FakeLogger:
    return FakeLogger()


@pytest.fixture
def client(settings: Settings, cache: InMemoryCache, logger: FakeLogger) -> USDAClient:
    return USDAClient(settings=settings, cache=cache, logger=logger)


# ---------------------------------------------------------------------------
# search() — happy path
# ---------------------------------------------------------------------------


@respx.mock
async def test_search_returns_candidates(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=SEARCH_RESPONSE)
    )

    results = await client.search("chicken breast")

    assert len(results) == 2
    first = results[0]
    assert first.fdc_id == 2646170  # post-migration Foundation ID (>= 2M)
    assert first.description == "Chicken Breast, Raw"
    assert first.data_type == USDADataType.FOUNDATION
    assert first.brand_owner is None
    assert first.nutrition_per_100g is not None
    assert first.nutrition_per_100g.protein_g == 23.2
    assert first.nutrition_per_100g.fat_g == 1.24
    assert first.nutrition_per_100g.calories_kcal == 120.0


@respx.mock
async def test_search_empty_results(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=EMPTY_SEARCH)
    )

    results = await client.search("xyzzy_nonexistent_food")
    assert results == []


@respx.mock
async def test_search_data_type_filter_sent_in_params(client: USDAClient) -> None:
    route = respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=EMPTY_SEARCH)
    )

    await client.search("broccoli", data_type=USDADataType.FOUNDATION)

    assert route.called
    request = route.calls.last.request
    assert "dataType=Foundation" in str(request.url)


@respx.mock
async def test_search_page_size_sent_in_params(client: USDAClient) -> None:
    route = respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=EMPTY_SEARCH)
    )

    await client.search("broccoli", page_size=5)

    request = route.calls.last.request
    assert "pageSize=5" in str(request.url)


@respx.mock
async def test_search_api_key_sent_in_params(client: USDAClient) -> None:
    route = respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=EMPTY_SEARCH)
    )

    await client.search("banana")

    request = route.calls.last.request
    assert "api_key=test-key" in str(request.url)


# ---------------------------------------------------------------------------
# search() — cache behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_search_uses_cache_on_hit(
    client: USDAClient, cache: InMemoryCache
) -> None:
    # Seed cache with serialised search results
    cache_key = "usda:search:chicken breast:all:v1"
    # We store a list of dicts (JSON-round-tripped candidates)
    cached_data = [
        {
            "fdc_id": 12345,
            "description": "Cached Chicken",
            "data_type": "Foundation",
            "brand_owner": None,
            "nutrition_per_100g": {
                "calories_kcal": 100.0,
                "protein_g": 20.0,
                "fat_g": 2.0,
                "saturated_fat_g": 0.5,
                "carbohydrates_g": 0.0,
                "sugar_g": 0.0,
                "fiber_g": 0.0,
                "sodium_mg": 50.0,
            },
            "relevance_score": 999.0,
        }
    ]
    cache.set(cache_key, cached_data)

    # No HTTP route registered — if HTTP fires, respx will raise
    results = await client.search("chicken breast")

    assert len(results) == 1
    assert results[0].fdc_id == 12345
    assert results[0].description == "Cached Chicken"


@respx.mock
async def test_search_populates_cache_on_miss(
    client: USDAClient, cache: InMemoryCache
) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=SEARCH_RESPONSE)
    )

    await client.search("chicken breast")

    cache_key = "usda:search:chicken breast:all:v1"
    cached = cache.get(cache_key)
    assert cached is not None
    assert isinstance(cached, list)
    assert len(cached) == 2


@respx.mock
async def test_search_cache_key_includes_data_type(
    client: USDAClient, cache: InMemoryCache
) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=EMPTY_SEARCH)
    )

    await client.search("banana", data_type=USDADataType.FOUNDATION)

    # Cache key should include the data_type
    cache_key = "usda:search:banana:Foundation:v1"
    assert cache.get(cache_key) is not None


# ---------------------------------------------------------------------------
# search() — HTTP errors
# ---------------------------------------------------------------------------


@respx.mock
async def test_search_401_raises(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(401, json={"message": "API key required"})
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.search("chicken breast")
    assert exc_info.value.response.status_code == 401


@respx.mock
async def test_search_429_raises_and_logs(
    client: USDAClient, logger: FakeLogger
) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(429, json={"message": "Too many requests"})
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.search("chicken breast")

    assert exc_info.value.response.status_code == 429
    warnings = logger.at_level("warning")
    assert len(warnings) >= 1
    assert any("429" in str(w) or "rate" in str(w).lower() or "too many" in str(w).lower() for w in warnings)


@respx.mock
async def test_search_network_error_raises(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/foods/search").mock(side_effect=httpx.ConnectError("timeout"))

    with pytest.raises(httpx.ConnectError):
        await client.search("chicken breast")


# ---------------------------------------------------------------------------
# get_food() — happy path
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_food_returns_candidate(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/food/171477").mock(
        return_value=httpx.Response(200, json=FOOD_RESPONSE)
    )

    candidate = await client.get_food(171477)

    assert candidate.fdc_id == 171477
    assert candidate.description == "Chicken, broilers or fryers, breast, meat only, raw"
    assert candidate.data_type == USDADataType.SR_LEGACY
    assert candidate.brand_owner is None


@respx.mock
async def test_get_food_nutrient_mapping(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/food/171477").mock(
        return_value=httpx.Response(200, json=FOOD_RESPONSE)
    )

    candidate = await client.get_food(171477)

    assert candidate.nutrition_per_100g is not None
    n = candidate.nutrition_per_100g
    assert n.protein_g == 23.2
    assert n.fat_g == 1.24
    assert n.carbohydrates_g == 0.0
    assert n.calories_kcal == 120.0
    assert n.fiber_g == 0.0
    assert n.sugar_g == 0.0
    assert n.sodium_mg == 69.0
    assert n.saturated_fat_g == 0.32


@respx.mock
async def test_get_food_missing_nutrients_defaults_to_zero(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/food/999999").mock(
        return_value=httpx.Response(200, json=MISSING_NUTRIENTS)
    )

    candidate = await client.get_food(999999)

    assert candidate.nutrition_per_100g is not None
    n = candidate.nutrition_per_100g
    assert n.protein_g == 0.0
    assert n.fat_g == 0.0
    assert n.calories_kcal == 0.0
    assert n.sodium_mg == 0.0


# ---------------------------------------------------------------------------
# get_food() — cache behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_food_uses_cache(
    client: USDAClient, cache: InMemoryCache
) -> None:
    cache_key = "usda:food:171477:v1"
    cached_data = {
        "fdc_id": 171477,
        "description": "Cached description",
        "data_type": "Foundation",
        "brand_owner": None,
        "nutrition_per_100g": {
            "calories_kcal": 200.0,
            "protein_g": 10.0,
            "fat_g": 5.0,
            "saturated_fat_g": 1.0,
            "carbohydrates_g": 20.0,
            "sugar_g": 5.0,
            "fiber_g": 2.0,
            "sodium_mg": 100.0,
        },
        "relevance_score": None,
    }
    cache.set(cache_key, cached_data)

    # No HTTP route — respx would error if HTTP fired
    candidate = await client.get_food(171477)

    assert candidate.fdc_id == 171477
    assert candidate.description == "Cached description"


@respx.mock
async def test_get_food_populates_cache_on_miss(
    client: USDAClient, cache: InMemoryCache
) -> None:
    respx.get(f"{BASE_URL}/food/171477").mock(
        return_value=httpx.Response(200, json=FOOD_RESPONSE)
    )

    await client.get_food(171477)

    assert cache.get("usda:food:171477:v1") is not None


# ---------------------------------------------------------------------------
# get_food() — HTTP errors
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_food_404_raises(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/food/1").mock(
        return_value=httpx.Response(404, json={"message": "Not found"})
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.get_food(1)
    assert exc_info.value.response.status_code == 404


@respx.mock
async def test_get_food_401_raises(client: USDAClient) -> None:
    respx.get(f"{BASE_URL}/food/171477").mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.get_food(171477)
    assert exc_info.value.response.status_code == 401


# ---------------------------------------------------------------------------
# Foundation ID validation — _is_likely_valid()
# ---------------------------------------------------------------------------


def test_is_likely_valid_drops_superseded_foundation_ids() -> None:
    """Pre-migration Foundation IDs (< 2M) must be rejected."""
    nutrition = NutritionPer100g(
        calories_kcal=0, protein_g=0, fat_g=0, saturated_fat_g=0,
        carbohydrates_g=0, sugar_g=0, fiber_g=0, sodium_mg=0,
    )
    superseded_ids = [746782, 747447, 748967]
    for fdc_id in superseded_ids:
        candidate = USDACandidate(
            fdc_id=fdc_id,
            description="Superseded",
            data_type=USDADataType.FOUNDATION,
            nutrition_per_100g=nutrition,
        )
        assert not _is_likely_valid(candidate), f"Expected {fdc_id} to be invalid"


def test_is_likely_valid_keeps_sr_legacy_regardless_of_id() -> None:
    """SR Legacy IDs in any range must NOT be filtered."""
    nutrition = NutritionPer100g(
        calories_kcal=0, protein_g=0, fat_g=0, saturated_fat_g=0,
        carbohydrates_g=0, sugar_g=0, fiber_g=0, sodium_mg=0,
    )
    for fdc_id in [746782, 747447, 748967, 171477]:
        candidate = USDACandidate(
            fdc_id=fdc_id,
            description="SR Legacy",
            data_type=USDADataType.SR_LEGACY,
            nutrition_per_100g=nutrition,
        )
        assert _is_likely_valid(candidate), f"Expected SR Legacy {fdc_id} to be valid"


def test_is_likely_valid_keeps_post_migration_foundation_ids() -> None:
    """Post-migration Foundation IDs (>= 2M) must pass through."""
    nutrition = NutritionPer100g(
        calories_kcal=0, protein_g=0, fat_g=0, saturated_fat_g=0,
        carbohydrates_g=0, sugar_g=0, fiber_g=0, sodium_mg=0,
    )
    for fdc_id in [2_000_000, 2_646_170, 2_708_638]:
        candidate = USDACandidate(
            fdc_id=fdc_id,
            description="Valid Foundation",
            data_type=USDADataType.FOUNDATION,
            nutrition_per_100g=nutrition,
        )
        assert _is_likely_valid(candidate), f"Expected {fdc_id} to be valid"


@respx.mock
async def test_search_filters_superseded_foundation_ids(client: USDAClient) -> None:
    """search() silently drops Foundation hits with pre-migration FDC IDs."""
    mixed_response = {
        "totalHits": 3,
        "currentPage": 1,
        "foods": [
            {
                "fdcId": 746782,         # superseded Foundation — must be dropped
                "description": "Chicken Breast, superseded",
                "dataType": "Foundation",
                "brandOwner": None,
                "score": 999.0,
                "foodNutrients": [],
            },
            {
                "fdcId": 2646170,        # valid post-migration Foundation — keep
                "description": "Chicken Breast, valid Foundation",
                "dataType": "Foundation",
                "brandOwner": None,
                "score": 950.0,
                "foodNutrients": [],
            },
            {
                "fdcId": 171477,         # SR Legacy — always keep
                "description": "Chicken, broilers, breast, raw",
                "dataType": "SR Legacy",
                "brandOwner": None,
                "score": 900.0,
                "foodNutrients": [],
            },
        ],
    }
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=mixed_response)
    )

    results = await client.search("chicken breast")

    fdc_ids = [r.fdc_id for r in results]
    assert 746782 not in fdc_ids, "Superseded Foundation ID must be filtered"
    assert 2646170 in fdc_ids, "Valid Foundation ID must be kept"
    assert 171477 in fdc_ids, "SR Legacy ID must be kept"
    assert len(results) == 2


@respx.mock
async def test_search_returns_none_nutrition_for_incomplete_inline_nutrients(
    client: USDAClient,
) -> None:
    """nutrition_per_100g is None when the search hit's foodNutrients is incomplete."""
    partial_response = {
        "totalHits": 1,
        "currentPage": 1,
        "foods": [
            {
                "fdcId": 2646170,
                "description": "Chicken Breast",
                "dataType": "Foundation",
                "brandOwner": None,
                "score": 900.0,
                "foodNutrients": [
                    # Only protein and fat — missing 6 of the 8 required IDs
                    {"nutrientId": 1003, "nutrientName": "Protein", "value": 22.5, "unitName": "G"},
                    {"nutrientId": 1004, "nutrientName": "Total lipid (fat)", "value": 1.93, "unitName": "G"},
                ],
            }
        ],
    }
    respx.get(f"{BASE_URL}/foods/search").mock(
        return_value=httpx.Response(200, json=partial_response)
    )

    results = await client.search("chicken breast")

    assert len(results) == 1
    assert results[0].nutrition_per_100g is None, (
        "Should be None — incomplete inline nutrients signal caller to use get_food()"
    )
