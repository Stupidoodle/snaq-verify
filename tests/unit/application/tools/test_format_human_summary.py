"""Table-driven tests for format_human_summary."""

import pytest

from snaq_verify.application.tools.format_human_summary import format_human_summary
from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.food_item import DefaultPortion, FoodItem, NutritionPer100g
from snaq_verify.domain.models.nutrient_comparison import (
    ItemVerdictBundle,
    NutrientDelta,
    NutrientVerdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nutrition(**kwargs: float) -> NutritionPer100g:
    defaults: dict[str, float] = dict(
        calories_kcal=100,
        protein_g=5,
        fat_g=3,
        saturated_fat_g=1,
        carbohydrates_g=10,
        sugar_g=2,
        fiber_g=1,
        sodium_mg=50,
    )
    defaults.update(kwargs)
    return NutritionPer100g(**defaults)


def _item(name: str = "Chicken Breast", brand: str | None = None) -> FoodItem:
    return FoodItem(
        id="test",
        name=name,
        brand=brand,
        default_portion=DefaultPortion(amount=100, unit="g", description="100g"),
        nutrition_per_100g=_nutrition(),
    )


def _verdict_nutrient(
    nutrient: str = "protein_g",
    reported: float = 10.0,
    observed: float = 10.0,
    verdict: Verdict = Verdict.MATCH,
) -> NutrientVerdict:
    return NutrientVerdict(
        nutrient=nutrient,
        delta=NutrientDelta(
            nutrient=nutrient,
            reported=reported,
            observed=observed,
            absolute_delta=abs(reported - observed),
            relative_delta_pct=(reported - observed) / observed * 100 if observed > 0 else None,
        ),
        verdict=verdict,
    )


def _bundle(
    nutrients: list[NutrientVerdict] | None = None,
    item_verdict: Verdict = Verdict.MATCH,
) -> ItemVerdictBundle:
    return ItemVerdictBundle(
        per_nutrient=nutrients or [],
        item_verdict=item_verdict,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_string() -> None:
    result = format_human_summary(_item(), _bundle(), evidence_count=2)
    assert isinstance(result, str)


def test_non_empty_string() -> None:
    result = format_human_summary(_item(), _bundle(), evidence_count=2)
    assert len(result) > 0


def test_item_name_in_output() -> None:
    result = format_human_summary(_item(name="Broccoli"), _bundle(), evidence_count=1)
    assert "Broccoli" in result


def test_brand_in_output_when_present() -> None:
    result = format_human_summary(_item(name="Greek Yogurt", brand="Fage"), _bundle(), evidence_count=3)
    assert "Fage" in result


def test_brand_absent_when_none() -> None:
    item = _item(name="Oats", brand=None)
    result = format_human_summary(item, _bundle(), evidence_count=2)
    # Should not raise; brand should not appear if None
    assert "None" not in result


def test_evidence_count_in_output() -> None:
    result = format_human_summary(_item(), _bundle(), evidence_count=3)
    assert "3" in result


@pytest.mark.parametrize("verdict, label", [
    (Verdict.MATCH, "match"),
    (Verdict.MINOR_DISCREPANCY, "minor"),
    (Verdict.MAJOR_DISCREPANCY, "major"),
])
def test_verdict_string_in_output(verdict: Verdict, label: str) -> None:
    bundle = _bundle(item_verdict=verdict)
    result = format_human_summary(_item(), bundle, evidence_count=2)
    assert label in result.lower()


def test_flagged_count_in_output_when_discrepancies() -> None:
    bundle = _bundle(
        nutrients=[
            _verdict_nutrient(verdict=Verdict.MAJOR_DISCREPANCY),
            _verdict_nutrient(nutrient="fat_g", verdict=Verdict.MINOR_DISCREPANCY),
            _verdict_nutrient(nutrient="fiber_g", verdict=Verdict.MATCH),
        ],
        item_verdict=Verdict.MAJOR_DISCREPANCY,
    )
    result = format_human_summary(_item(), bundle, evidence_count=2)
    # 2 nutrients are flagged (not match)
    assert "2" in result


def test_all_match_no_flagged_count_or_zero() -> None:
    bundle = _bundle(
        nutrients=[
            _verdict_nutrient(verdict=Verdict.MATCH),
            _verdict_nutrient(nutrient="fat_g", verdict=Verdict.MATCH),
        ],
        item_verdict=Verdict.MATCH,
    )
    result = format_human_summary(_item(), bundle, evidence_count=1)
    assert isinstance(result, str)
    # Should communicate all is well
    assert "match" in result.lower() or "verified" in result.lower() or "0" in result


def test_zero_evidence_count() -> None:
    result = format_human_summary(_item(), _bundle(), evidence_count=0)
    assert isinstance(result, str)
    assert "0" in result


def test_no_llm_call_pure_template() -> None:
    """Calling twice with same inputs must return identical output."""
    item = _item(name="Salmon", brand="Acme")
    bundle = _bundle(
        nutrients=[_verdict_nutrient(verdict=Verdict.MAJOR_DISCREPANCY)],
        item_verdict=Verdict.MAJOR_DISCREPANCY,
    )
    r1 = format_human_summary(item, bundle, evidence_count=3)
    r2 = format_human_summary(item, bundle, evidence_count=3)
    assert r1 == r2
