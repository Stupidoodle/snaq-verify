"""Tests for the search_usda tool factory."""

import pytest
from agents import FunctionTool

from snaq_verify.application.tools.search_usda import make_search_usda
from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.domain.models.source_lookup import USDACandidate
from tests.fakes.fake_usda_client import FakeUSDAClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nutrition(**kwargs: float) -> NutritionPer100g:
    defaults: dict[str, float] = dict(
        calories_kcal=120.0,
        protein_g=23.2,
        fat_g=1.24,
        saturated_fat_g=0.32,
        carbohydrates_g=0.0,
        sugar_g=0.0,
        fiber_g=0.0,
        sodium_mg=69.0,
    )
    defaults.update(kwargs)
    return NutritionPer100g(**defaults)


def _candidate(
    fdc_id: int = 171477,
    description: str = "Chicken, breast, raw",
    data_type: USDADataType = USDADataType.SR_LEGACY,
) -> USDACandidate:
    return USDACandidate(
        fdc_id=fdc_id,
        description=description,
        data_type=data_type,
        nutrition_per_100g=_nutrition(),
        relevance_score=987.3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_search_usda_returns_candidates() -> None:
    fake = FakeUSDAClient()
    fake.add_search_result("chicken breast", [_candidate()])
    fn, _ = make_search_usda(fake)

    results = await fn("chicken breast")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["fdc_id"] == 171477
    assert results[0]["description"] == "Chicken, breast, raw"


async def test_search_usda_empty_returns_empty_list() -> None:
    fake = FakeUSDAClient()  # no results registered
    fn, _ = make_search_usda(fake)

    results = await fn("xyzzy_food_does_not_exist")

    assert results == []


async def test_search_usda_passes_data_type_filter() -> None:
    fake = FakeUSDAClient()
    fake.add_search_result("banana", [_candidate(fdc_id=999, description="Banana, raw")])
    fn, _ = make_search_usda(fake)

    results = await fn("banana", data_type="Foundation")

    assert fake.search_calls[0]["data_type"] == USDADataType.FOUNDATION
    assert len(results) == 1


async def test_search_usda_passes_page_size() -> None:
    fake = FakeUSDAClient()
    fake.add_search_result("oats", [_candidate(fdc_id=123, description="Oats, rolled")])
    fn, _ = make_search_usda(fake)

    await fn("oats", page_size=5)

    assert fake.search_calls[0]["page_size"] == 5


async def test_search_usda_none_data_type_passes_none() -> None:
    fake = FakeUSDAClient()
    fake.add_search_result("broccoli", [_candidate(fdc_id=456, description="Broccoli")])
    fn, _ = make_search_usda(fake)

    await fn("broccoli")

    assert fake.search_calls[0]["data_type"] is None


async def test_search_usda_result_has_nutrition_fields() -> None:
    fake = FakeUSDAClient()
    fake.add_search_result("chicken", [_candidate()])
    fn, _ = make_search_usda(fake)

    results = await fn("chicken")

    n = results[0]["nutrition_per_100g"]
    assert n["protein_g"] == 23.2
    assert n["sodium_mg"] == 69.0


async def test_search_usda_multiple_results_preserved_in_order() -> None:
    fake = FakeUSDAClient()
    candidates = [
        _candidate(fdc_id=100, description="First"),
        _candidate(fdc_id=200, description="Second"),
        _candidate(fdc_id=300, description="Third"),
    ]
    fake.add_search_result("multi", candidates)
    fn, _ = make_search_usda(fake)

    results = await fn("multi")

    assert [r["fdc_id"] for r in results] == [100, 200, 300]


async def test_search_usda_propagates_error() -> None:
    fake = FakeUSDAClient(raise_on="bad_query")
    fn, _ = make_search_usda(fake)

    with pytest.raises(RuntimeError, match="simulated error"):
        await fn("bad_query")


def test_tool_export_is_function_tool() -> None:
    """Factory exports a FunctionTool suitable for Agent(tools=[...])."""
    fake = FakeUSDAClient()
    _, tool = make_search_usda(fake)

    assert isinstance(tool, FunctionTool)
    assert tool.name == "search_usda"
