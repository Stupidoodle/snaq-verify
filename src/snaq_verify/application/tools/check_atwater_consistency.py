"""Atwater equation sanity check for a nutrition entry."""

from snaq_verify.domain.models.atwater_check import AtwaterCheck
from snaq_verify.domain.models.food_item import NutritionPer100g

# Atwater factors (kcal per gram of macronutrient).
ATWATER_PROTEIN: float = 4.0
ATWATER_CARB: float = 4.0
ATWATER_FAT: float = 9.0


def check_atwater_consistency(
    nutrition: NutritionPer100g,
    tolerance_pct: float,
) -> AtwaterCheck:
    """Cross-check ``calories_kcal`` against the Atwater equation.

    Expected kcal = 4 × protein_g + 4 × carbohydrates_g + 9 × fat_g.

    A significant deviation between the reported calorie value and the
    Atwater prediction signals that macros and kcal were drawn from different
    sources, or one field is wrong.

    The relative delta uses ``max(reported_kcal, 1)`` as the denominator to
    avoid division by zero when ``calories_kcal == 0``.

    Args:
        nutrition: The nutrition payload to check.
        tolerance_pct: Maximum acceptable relative delta (%) for consistency.

    Returns:
        :class:`AtwaterCheck` with all computed fields and an
        ``is_consistent`` flag.
    """
    expected_kcal = (
        ATWATER_PROTEIN * nutrition.protein_g
        + ATWATER_CARB * nutrition.carbohydrates_g
        + ATWATER_FAT * nutrition.fat_g
    )
    reported_kcal = nutrition.calories_kcal
    absolute_delta = abs(reported_kcal - expected_kcal)
    relative_delta_pct = absolute_delta / max(reported_kcal, 1.0) * 100.0
    is_consistent = relative_delta_pct <= tolerance_pct

    return AtwaterCheck(
        nutrition=nutrition,
        expected_kcal=expected_kcal,
        reported_kcal=reported_kcal,
        absolute_delta=absolute_delta,
        relative_delta_pct=relative_delta_pct,
        is_consistent=is_consistent,
    )
