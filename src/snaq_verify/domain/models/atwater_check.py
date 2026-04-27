"""Atwater-equation sanity check model."""

from pydantic import BaseModel

from snaq_verify.domain.models.food_item import NutritionPer100g


class AtwaterCheck(BaseModel):
    """Cross-check `kcal_reported` vs `4*protein + 4*carbs + 9*fat`.

    A stable identity for whole-food entries — large deviations indicate the
    macros and the kcal were filled from different sources or one is wrong.
    """

    nutrition: NutritionPer100g
    expected_kcal: float
    reported_kcal: float
    absolute_delta: float
    relative_delta_pct: float
    is_consistent: bool
