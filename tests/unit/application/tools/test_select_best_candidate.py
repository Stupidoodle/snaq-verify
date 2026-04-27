"""Table-driven tests for select_best_candidate."""

import pytest

from snaq_verify.application.tools.select_best_candidate import select_best_candidate
from snaq_verify.domain.models.food_item import DefaultPortion, FoodItem, NutritionPer100g
from snaq_verify.domain.models.source_lookup import SelectedCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nutrition() -> NutritionPer100g:
    return NutritionPer100g(
        calories_kcal=100,
        protein_g=5,
        fat_g=3,
        saturated_fat_g=1,
        carbohydrates_g=10,
        sugar_g=2,
        fiber_g=1,
        sodium_mg=50,
    )


def _item(name: str = "Chicken Breast", brand: str | None = None) -> FoodItem:
    return FoodItem(
        id="test",
        name=name,
        brand=brand,
        default_portion=DefaultPortion(amount=100, unit="g", description="100g"),
        nutrition_per_100g=_nutrition(),
    )


def _candidate(
    source: str = "usda",
    source_id: str = "100",
    source_name: str = "Chicken Breast",
) -> SelectedCandidate:
    return SelectedCandidate(
        source=source,
        source_id=source_id,
        source_name=source_name,
        nutrition_per_100g=_nutrition(),
        match_score=0.9,
    )


# ---------------------------------------------------------------------------
# Parametrized cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "candidates, min_score, expect_none",
    [
        # Nominal: single candidate well above threshold
        ([_candidate(source_name="Chicken Breast")], 0.5, False),
        # Empty list → None
        ([], 0.5, True),
        # min_score = 1.0 → nothing can reach it normally
        ([_candidate(source_name="Totally Different Food Xyz")], 1.0, True),
        # min_score = 0.0 → always picks the best
        ([_candidate(source_name="Anything Here")], 0.0, False),
    ],
)
def test_basic_cases(
    candidates: list[SelectedCandidate],
    min_score: float,
    expect_none: bool,
) -> None:
    item = _item(name="Chicken Breast")
    result = select_best_candidate(item, candidates, min_score)
    if expect_none:
        assert result is None
    else:
        assert result is not None
        assert isinstance(result, SelectedCandidate)


def test_picks_highest_scored_candidate() -> None:
    item = _item(name="Banana")
    # "Banana" exact match beats "Banana Chips" which has extra token
    exact = _candidate(source="usda", source_id="200", source_name="Banana")
    partial = _candidate(source="usda", source_id="300", source_name="Banana Chips Processed")
    result = select_best_candidate(item, [partial, exact], min_score=0.0)
    assert result is not None
    assert result.source_id == "200"


def test_deterministic_tie_break_lowest_source_id() -> None:
    """When two candidates score equally, lowest source_id wins."""
    item = _item(name="Banana")
    c1 = _candidate(source="usda", source_id="AAA111", source_name="Banana")
    c2 = _candidate(source="usda", source_id="AAA222", source_name="Banana")
    # Both have same name and source → same score
    r1 = select_best_candidate(item, [c1, c2], min_score=0.0)
    r2 = select_best_candidate(item, [c2, c1], min_score=0.0)
    assert r1 is not None
    assert r2 is not None
    # Must be deterministic regardless of input order
    assert r1.source_id == r2.source_id == "AAA111"


def test_below_min_score_returns_none() -> None:
    item = _item(name="Broccoli")
    # Completely unrelated candidate
    candidate = _candidate(source="web", source_name="Chocolate Cake Frosted")
    result = select_best_candidate(item, [candidate], min_score=0.5)
    assert result is None


def test_multiple_candidates_returns_best() -> None:
    item = _item(name="Oats")
    good = _candidate(source="usda", source_id="10", source_name="Oats")
    bad = _candidate(source="web", source_id="20", source_name="Chocolate Cake Frosted")
    result = select_best_candidate(item, [bad, good], min_score=0.0)
    assert result is not None
    assert result.source_id == "10"


def test_single_candidate_exactly_at_threshold() -> None:
    """A candidate that exactly meets min_score should be returned."""
    item = _item(name="Banana")
    candidate = _candidate(source="usda", source_id="1", source_name="Banana")
    from snaq_verify.application.tools.score_candidate_match import score_candidate_match
    actual_score = score_candidate_match(item, candidate)
    # Use actual score as threshold → should be selected
    result = select_best_candidate(item, [candidate], min_score=actual_score)
    assert result is not None


def test_returns_none_for_empty_candidates() -> None:
    item = _item()
    assert select_best_candidate(item, [], min_score=0.0) is None
