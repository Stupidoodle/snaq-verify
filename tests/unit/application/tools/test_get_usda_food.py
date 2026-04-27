"""Tests for the get_usda_food tool factory."""

import pytest
from agents import FunctionTool

from snaq_verify.application.tools.get_usda_food import make_get_usda_food
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
    description: str = "Chicken, broilers, breast, raw",
    data_type: USDADataType = USDADataType.SR_LEGACY,
) -> USDACandidate:
    return USDACandidate(
        fdc_id=fdc_id,
        description=description,
        data_type=data_type,
        nutrition_per_100g=_nutrition(),
        relevance_score=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_get_usda_food_returns_candidate_dict() -> None:
    fake = FakeUSDAClient()
    fake.add_food(171477, _candidate())
    fn, _ = make_get_usda_food(fake)

    result = await fn(171477)

    assert isinstance(result, dict)
    assert result["fdc_id"] == 171477
    assert result["description"] == "Chicken, broilers, breast, raw"


async def test_get_usda_food_nutrition_fields_present() -> None:
    fake = FakeUSDAClient()
    fake.add_food(171477, _candidate())
    fn, _ = make_get_usda_food(fake)

    result = await fn(171477)

    n = result["nutrition_per_100g"]
    assert n["protein_g"] == 23.2
    assert n["calories_kcal"] == 120.0
    assert n["sodium_mg"] == 69.0
    assert n["fat_g"] == 1.24


async def test_get_usda_food_data_type_serialized() -> None:
    fake = FakeUSDAClient()
    fake.add_food(100, _candidate(fdc_id=100, data_type=USDADataType.FOUNDATION))
    fn, _ = make_get_usda_food(fake)

    result = await fn(100)

    assert result["data_type"] == "Foundation"


async def test_get_usda_food_propagates_not_found_error() -> None:
    fake = FakeUSDAClient()  # nothing registered
    fn, _ = make_get_usda_food(fake)

    with pytest.raises(KeyError):
        await fn(999999)


async def test_get_usda_food_propagates_simulated_error() -> None:
    fake = FakeUSDAClient(raise_on_fdc_id=99)
    fn, _ = make_get_usda_food(fake)

    with pytest.raises(RuntimeError, match="simulated error"):
        await fn(99)


async def test_get_usda_food_records_call() -> None:
    fake = FakeUSDAClient()
    fake.add_food(171477, _candidate())
    fn, _ = make_get_usda_food(fake)

    await fn(171477)

    assert 171477 in fake.get_food_calls


def test_tool_export_is_function_tool() -> None:
    """Factory exports a FunctionTool suitable for Agent(tools=[...])."""
    fake = FakeUSDAClient()
    _, tool = make_get_usda_food(fake)

    assert isinstance(tool, FunctionTool)
    assert tool.name == "get_usda_food"
