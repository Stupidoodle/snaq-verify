"""Compute per-nutrient deltas between reported and observed nutrition."""

from agents import function_tool

from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.domain.models.nutrient_comparison import NutrientDelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Below this absolute observed value we suppress relative-delta computation
#: to avoid false majors on near-zero nutrients (e.g., 0.0 g vs 0.3 g fiber).
ABSOLUTE_FLOOR: float = 0.5

#: Ordered list of ``NutritionPer100g`` field names to compare.
NUTRIENT_FIELDS: tuple[str, ...] = (
    "calories_kcal",
    "protein_g",
    "fat_g",
    "saturated_fat_g",
    "carbohydrates_g",
    "sugar_g",
    "fiber_g",
    "sodium_mg",
)


def compute_per_nutrient_delta(
    reported: NutritionPer100g,
    observed: NutritionPer100g,
) -> list[NutrientDelta]:
    """Compute a signed delta for every nutrient field.

    For each nutrient in :data:`NUTRIENT_FIELDS`:

    - ``absolute_delta = |reported − observed|`` — always computed.
    - ``relative_delta_pct = (reported − observed) / observed * 100`` — set to
      ``None`` when ``observed < ABSOLUTE_FLOOR`` to suppress noise on
      near-zero values.

    Args:
        reported: Nutrition values as stated on the label / input file.
        observed: Nutrition values from the external source (USDA / OFF / web).

    Returns:
        One :class:`~snaq_verify.domain.models.nutrient_comparison.NutrientDelta`
        per field in :data:`NUTRIENT_FIELDS`, preserving field order.
    """
    deltas: list[NutrientDelta] = []
    for field in NUTRIENT_FIELDS:
        r: float = getattr(reported, field)
        o: float = getattr(observed, field)
        absolute_delta = abs(r - o)
        relative_delta_pct: float | None = (
            (r - o) / o * 100 if o >= ABSOLUTE_FLOOR else None
        )
        deltas.append(
            NutrientDelta(
                nutrient=field,
                reported=r,
                observed=o,
                absolute_delta=absolute_delta,
                relative_delta_pct=relative_delta_pct,
            )
        )
    return deltas


compute_per_nutrient_delta_tool = function_tool(compute_per_nutrient_delta)
