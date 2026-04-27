"""Table-driven tests for check_atwater_consistency."""

import pytest

from snaq_verify.application.tools.check_atwater_consistency import check_atwater_consistency
from snaq_verify.domain.models.atwater_check import AtwaterCheck
from snaq_verify.domain.models.food_item import NutritionPer100g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nutrition(
    calories_kcal: float = 200.0,
    protein_g: float = 10.0,
    fat_g: float = 5.0,
    carbohydrates_g: float = 20.0,
    **kwargs: float,
) -> NutritionPer100g:
    return NutritionPer100g(
        calories_kcal=calories_kcal,
        protein_g=protein_g,
        fat_g=fat_g,
        saturated_fat_g=kwargs.get("saturated_fat_g", 1.0),
        carbohydrates_g=carbohydrates_g,
        sugar_g=kwargs.get("sugar_g", 2.0),
        fiber_g=kwargs.get("fiber_g", 1.0),
        sodium_mg=kwargs.get("sodium_mg", 50.0),
    )


def _expected_kcal(protein_g: float, fat_g: float, carbohydrates_g: float) -> float:
    return 4.0 * protein_g + 4.0 * carbohydrates_g + 9.0 * fat_g


# ---------------------------------------------------------------------------
# Parametrized cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "protein_g, fat_g, carbs_g, reported_kcal, tolerance_pct, expect_consistent",
    [
        # Nominal: exact Atwater match
        (10.0, 5.0, 20.0, 4*10+9*5+4*20, 15.0, True),   # exact
        # Within tolerance (5% < 15%)
        (10.0, 5.0, 20.0, 4*10+9*5+4*20 * 1.05, 15.0, True),
        # Exactly at tolerance boundary
        (10.0, 5.0, 20.0, (4*10+9*5+4*20) * (1 + 15/100), 15.0, True),
        # Just over tolerance
        (10.0, 5.0, 20.0, (4*10+9*5+4*20) * 1.20, 15.0, False),
        # Large discrepancy (Fage 0% protein entry error simulation)
        (5.5, 0.0, 4.0, 97.0, 15.0, False),
        # All macros zero: expected_kcal=0, reported=0 → consistent
        (0.0, 0.0, 0.0, 0.0, 15.0, True),
        # All macros zero but reported > 0: should be inconsistent
        (0.0, 0.0, 0.0, 100.0, 15.0, False),
        # Very tight tolerance with perfect match
        (20.0, 10.0, 30.0, 4*20+9*10+4*30, 0.0, True),
        # Very tight tolerance with tiny deviation
        (20.0, 10.0, 30.0, 4*20+9*10+4*30 + 1.0, 0.0, False),
    ],
)
def test_is_consistent(
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    reported_kcal: float,
    tolerance_pct: float,
    expect_consistent: bool,
) -> None:
    n = NutritionPer100g(
        calories_kcal=reported_kcal,
        protein_g=protein_g,
        fat_g=fat_g,
        saturated_fat_g=0.0,
        carbohydrates_g=carbs_g,
        sugar_g=0.0,
        fiber_g=0.0,
        sodium_mg=0.0,
    )
    result = check_atwater_consistency(n, tolerance_pct)
    assert result.is_consistent == expect_consistent


# ---------------------------------------------------------------------------
# Formula accuracy
# ---------------------------------------------------------------------------

def test_expected_kcal_formula() -> None:
    """expected_kcal = 4*protein + 4*carbs + 9*fat."""
    n = _nutrition(protein_g=10, fat_g=5, carbohydrates_g=20, calories_kcal=200)
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    expected = 4 * 10 + 9 * 5 + 4 * 20
    assert result.expected_kcal == pytest.approx(expected)


def test_reported_kcal_matches_input() -> None:
    n = _nutrition(calories_kcal=350.0)
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    assert result.reported_kcal == pytest.approx(350.0)


def test_absolute_delta_is_non_negative() -> None:
    n = _nutrition()
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    assert result.absolute_delta >= 0.0


def test_relative_delta_pct_non_negative() -> None:
    n = _nutrition()
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    assert result.relative_delta_pct >= 0.0


def test_relative_delta_pct_formula_reported_gt_0() -> None:
    """relative_delta_pct = |reported - expected| / max(reported, 1) * 100."""
    protein_g, fat_g, carbs_g = 10.0, 5.0, 20.0
    expected = 4 * protein_g + 9 * fat_g + 4 * carbs_g
    reported = expected + 20.0  # 20 kcal off
    n = NutritionPer100g(
        calories_kcal=reported,
        protein_g=protein_g,
        fat_g=fat_g,
        saturated_fat_g=0.0,
        carbohydrates_g=carbs_g,
        sugar_g=0.0,
        fiber_g=0.0,
        sodium_mg=0.0,
    )
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    expected_pct = abs(reported - expected) / max(reported, 1) * 100
    assert result.relative_delta_pct == pytest.approx(expected_pct, rel=1e-6)


# ---------------------------------------------------------------------------
# Degenerate / edge
# ---------------------------------------------------------------------------

def test_returns_atwater_check() -> None:
    n = _nutrition()
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    assert isinstance(result, AtwaterCheck)


def test_nutrition_field_preserved() -> None:
    n = _nutrition()
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    assert result.nutrition == n


def test_zero_reported_kcal_no_division_by_zero() -> None:
    """When reported_kcal=0, max(reported, 1) prevents division by zero."""
    n = NutritionPer100g(
        calories_kcal=0.0,
        protein_g=0.0,
        fat_g=0.0,
        saturated_fat_g=0.0,
        carbohydrates_g=0.0,
        sugar_g=0.0,
        fiber_g=0.0,
        sodium_mg=0.0,
    )
    result = check_atwater_consistency(n, tolerance_pct=15.0)
    assert result.relative_delta_pct == pytest.approx(0.0)
    assert result.is_consistent is True
