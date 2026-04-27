"""Table-driven tests for score_candidate_match."""

import pytest

from snaq_verify.application.tools.score_candidate_match import score_candidate_match
from snaq_verify.domain.models.food_item import DefaultPortion, FoodItem, NutritionPer100g
from snaq_verify.domain.models.source_lookup import SelectedCandidate


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


def _item(
    name: str = "Chicken Breast",
    brand: str | None = None,
    category: str | None = None,
) -> FoodItem:
    return FoodItem(
        id="test",
        name=name,
        brand=brand,
        category=category,
        default_portion=DefaultPortion(amount=100, unit="g", description="100g serving"),
        nutrition_per_100g=_nutrition(),
    )


def _candidate(
    source: str = "usda",
    source_id: str = "171477",
    source_name: str = "Chicken Breast Raw",
    match_score: float = 0.9,
) -> SelectedCandidate:
    return SelectedCandidate(
        source=source,
        source_id=source_id,
        source_name=source_name,
        nutrition_per_100g=_nutrition(),
        match_score=match_score,
    )


# ---------------------------------------------------------------------------
# Parametrized range tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, source_name, brand, category, source, lo, hi",
    [
        # Perfect name match, USDA source
        ("Chicken Breast", "Chicken Breast", None, None, "usda", 0.65, 1.0),
        # No name overlap at all
        ("Banana", "Chicken Breast", None, None, "usda", 0.0, 0.25),
        # Partial name overlap (2/5 tokens intersect)
        ("Chicken Breast Cooked", "Chicken Breast Raw", None, None, "usda", 0.3, 0.8),
        # USDA source prior (high)
        ("Banana", "Banana", None, None, "usda", 0.65, 1.0),
        # OFF source prior (medium)
        ("Banana", "Banana", None, None, "off", 0.60, 0.95),
        # Web source prior (lowest)
        ("Banana", "Banana", None, None, "web", 0.55, 0.90),
        # Brand match contained in source_name
        ("Fage Greek Yogurt", "Fage Total 0% Plain", "Fage", None, "usda", 0.5, 1.0),
        # Empty item name → only source prior contributes
        ("", "Chicken Breast", None, None, "usda", 0.0, 0.25),
        # Identical names, category present in source_name
        ("Salmon", "Salmon Fillet", "Acme", "seafood", "usda", 0.6, 1.0),
        # Degenerate: empty source_name
        ("Chicken Breast", "", None, None, "usda", 0.0, 0.25),
    ],
)
def test_score_range(
    name: str,
    source_name: str,
    brand: str | None,
    category: str | None,
    source: str,
    lo: float,
    hi: float,
) -> None:
    item = _item(name=name, brand=brand, category=category)
    candidate = _candidate(source=source, source_name=source_name)
    score = score_candidate_match(item, candidate)
    assert 0.0 <= score <= 1.0, "Score must be in [0, 1]"
    assert lo <= score <= hi, f"Expected [{lo}, {hi}], got {score:.4f}"


# ---------------------------------------------------------------------------
# Ordering invariants
# ---------------------------------------------------------------------------

def test_usda_beats_off_same_name() -> None:
    item = _item(name="Banana")
    usda = _candidate(source="usda", source_name="Banana")
    off = _candidate(source="off", source_name="Banana")
    assert score_candidate_match(item, usda) > score_candidate_match(item, off)


def test_off_beats_web_same_name() -> None:
    item = _item(name="Banana")
    off = _candidate(source="off", source_name="Banana")
    web = _candidate(source="web", source_name="Banana")
    assert score_candidate_match(item, off) > score_candidate_match(item, web)


def test_brand_match_raises_score() -> None:
    item = _item(name="Greek Yogurt", brand="Fage")
    with_brand = _candidate(source_name="Fage Greek Yogurt")
    without_brand = _candidate(source_name="Greek Yogurt")
    assert score_candidate_match(item, with_brand) > score_candidate_match(item, without_brand)


def test_category_match_raises_score() -> None:
    item = _item(name="Salmon", category="seafood")
    with_cat = _candidate(source_name="Salmon seafood fillet")
    without_cat = _candidate(source_name="Salmon fillet")
    assert score_candidate_match(item, with_cat) >= score_candidate_match(item, without_cat)


# ---------------------------------------------------------------------------
# Boundary and degenerate cases
# ---------------------------------------------------------------------------

def test_returns_float() -> None:
    item = _item()
    candidate = _candidate()
    score = score_candidate_match(item, candidate)
    assert isinstance(score, float)


def test_identical_names_max_name_contribution() -> None:
    """Identical names should produce Jaccard = 1.0 (max name contribution)."""
    item = _item(name="Oat Meal")
    candidate = _candidate(source_name="Oat Meal", source="usda")
    score = score_candidate_match(item, candidate)
    # W_NAME * 1.0 + W_SOURCE * 1.0 ≥ 0.7
    assert score >= 0.65


def test_completely_disjoint_names_low_score() -> None:
    item = _item(name="Broccoli")
    candidate = _candidate(source_name="Whole Milk Yogurt", source="web")
    score = score_candidate_match(item, candidate)
    # Only source prior (web = lowest) contributes
    assert score <= 0.20


def test_none_brand_no_crash() -> None:
    """brand=None on FoodItem must not raise."""
    item = _item(name="Broccoli", brand=None)
    candidate = _candidate(source_name="Broccoli")
    score = score_candidate_match(item, candidate)
    assert 0.0 <= score <= 1.0
