"""Table-driven tests for compute_per_nutrient_delta."""

import pytest

from snaq_verify.application.tools.compute_per_nutrient_delta import (
    ABSOLUTE_FLOOR,
    NUTRIENT_FIELDS,
    compute_per_nutrient_delta,
)
from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.domain.models.nutrient_comparison import NutrientDelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nutrition(**kwargs: float) -> NutritionPer100g:
    defaults: dict[str, float] = dict(
        calories_kcal=200,
        protein_g=10,
        fat_g=5,
        saturated_fat_g=2,
        carbohydrates_g=20,
        sugar_g=4,
        fiber_g=3,
        sodium_mg=200,
    )
    defaults.update(kwargs)
    return NutritionPer100g(**defaults)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_returns_all_nutrient_fields() -> None:
    reported = _nutrition()
    observed = _nutrition()
    deltas = compute_per_nutrient_delta(reported, observed)
    assert len(deltas) == len(NUTRIENT_FIELDS)
    returned_names = {d.nutrient for d in deltas}
    assert returned_names == set(NUTRIENT_FIELDS)


def test_returns_list_of_nutrient_deltas() -> None:
    deltas = compute_per_nutrient_delta(_nutrition(), _nutrition())
    assert all(isinstance(d, NutrientDelta) for d in deltas)


# ---------------------------------------------------------------------------
# Nominal and identical cases
# ---------------------------------------------------------------------------

def test_identical_inputs_all_zero_deltas() -> None:
    n = _nutrition()
    deltas = compute_per_nutrient_delta(n, n)
    for d in deltas:
        assert d.absolute_delta == pytest.approx(0.0, abs=1e-9)
        if d.relative_delta_pct is not None:
            assert d.relative_delta_pct == pytest.approx(0.0, abs=1e-9)


def test_absolute_delta_correct() -> None:
    reported = _nutrition(protein_g=12.0)
    observed = _nutrition(protein_g=10.0)
    deltas = compute_per_nutrient_delta(reported, observed)
    protein_delta = next(d for d in deltas if d.nutrient == "protein_g")
    assert protein_delta.absolute_delta == pytest.approx(2.0)


def test_relative_delta_pct_formula() -> None:
    """relative_delta_pct = (reported - observed) / observed * 100."""
    reported = _nutrition(protein_g=12.0)
    observed = _nutrition(protein_g=10.0)
    deltas = compute_per_nutrient_delta(reported, observed)
    protein_delta = next(d for d in deltas if d.nutrient == "protein_g")
    # (12 - 10) / 10 * 100 = 20%
    assert protein_delta.relative_delta_pct == pytest.approx(20.0)


def test_negative_relative_delta() -> None:
    """Reported < observed → negative relative delta."""
    reported = _nutrition(protein_g=8.0)
    observed = _nutrition(protein_g=10.0)
    deltas = compute_per_nutrient_delta(reported, observed)
    protein_delta = next(d for d in deltas if d.nutrient == "protein_g")
    assert protein_delta.relative_delta_pct == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Floor suppression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "observed_fiber, expect_none",
    [
        (0.0, True),          # below floor → suppress
        (ABSOLUTE_FLOOR - 0.01, True),   # just below floor
        (ABSOLUTE_FLOOR, False),         # exactly at floor → not suppressed
        (ABSOLUTE_FLOOR + 0.01, False),  # just above floor
        (3.0, False),         # well above floor
    ],
)
def test_floor_suppression_relative_delta(observed_fiber: float, expect_none: bool) -> None:
    reported = _nutrition(fiber_g=1.0)
    observed = _nutrition(fiber_g=observed_fiber)
    deltas = compute_per_nutrient_delta(reported, observed)
    fiber_delta = next(d for d in deltas if d.nutrient == "fiber_g")
    if expect_none:
        assert fiber_delta.relative_delta_pct is None
    else:
        assert fiber_delta.relative_delta_pct is not None


def test_floor_suppression_does_not_affect_absolute_delta() -> None:
    """Even when relative_delta is suppressed, absolute_delta is still computed."""
    reported = _nutrition(fiber_g=0.3)
    observed = _nutrition(fiber_g=0.0)
    deltas = compute_per_nutrient_delta(reported, observed)
    fiber_delta = next(d for d in deltas if d.nutrient == "fiber_g")
    assert fiber_delta.absolute_delta == pytest.approx(0.3)
    assert fiber_delta.relative_delta_pct is None


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------

def test_large_values_no_overflow() -> None:
    reported = _nutrition(calories_kcal=900, protein_g=80, fat_g=80)
    observed = _nutrition(calories_kcal=100, protein_g=1, fat_g=1)
    deltas = compute_per_nutrient_delta(reported, observed)
    assert all(isinstance(d.absolute_delta, float) for d in deltas)


def test_reported_and_observed_fields_match() -> None:
    reported = _nutrition(protein_g=15.0, fat_g=7.0)
    observed = _nutrition(protein_g=10.0, fat_g=5.0)
    deltas = compute_per_nutrient_delta(reported, observed)
    protein = next(d for d in deltas if d.nutrient == "protein_g")
    fat = next(d for d in deltas if d.nutrient == "fat_g")
    assert protein.reported == pytest.approx(15.0)
    assert protein.observed == pytest.approx(10.0)
    assert fat.reported == pytest.approx(7.0)
    assert fat.observed == pytest.approx(5.0)


def test_sodium_floor_suppression() -> None:
    """sodium_mg: if observed < ABSOLUTE_FLOOR (0.5 mg), suppress relative delta."""
    reported = _nutrition(sodium_mg=1.0)
    observed = _nutrition(sodium_mg=0.1)
    deltas = compute_per_nutrient_delta(reported, observed)
    sodium_delta = next(d for d in deltas if d.nutrient == "sodium_mg")
    assert sodium_delta.relative_delta_pct is None
