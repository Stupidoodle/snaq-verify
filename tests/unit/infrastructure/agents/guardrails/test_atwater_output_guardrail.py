"""Unit tests for atwater_output_guardrail."""

from agents import RunContextWrapper, set_tracing_disabled

from snaq_verify.domain.models.atwater_check import AtwaterCheck
from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.infrastructure.agents.guardrails.atwater_output_guardrail import (
    atwater_output_guardrail,
)
from tests.fakes.factories import (
    make_atwater_check,
    make_food_item,
    make_item_verification,
)

set_tracing_disabled(True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CTX = RunContextWrapper(context=None)


def _nutrition(
    kcal: float = 100.0,
    protein: float = 10.0,
    fat: float = 2.0,
    carbs: float = 10.0,
) -> NutritionPer100g:
    """Return a NutritionPer100g with controlled macro values."""
    return NutritionPer100g(
        calories_kcal=kcal,
        protein_g=protein,
        fat_g=fat,
        saturated_fat_g=0.5,
        carbohydrates_g=carbs,
        sugar_g=1.0,
        fiber_g=0.5,
        sodium_mg=50.0,
    )


async def _run(n: NutritionPer100g, atwater: AtwaterCheck):  # type: ignore[no-untyped-def]
    """Build a verification with the given nutrition + atwater, run the guardrail."""
    item = make_food_item(nutrition=n)
    v = make_item_verification(item=item)
    v = v.model_copy(
        update={"reported_nutrition": n, "atwater_check_input": atwater},
    )
    return await atwater_output_guardrail.run(  # type: ignore[arg-type]
        context=_CTX, agent=None, agent_output=v,
    )


# ---------------------------------------------------------------------------
# Happy path — tripwire stays off
# ---------------------------------------------------------------------------


async def test_consistent_nutrition_no_tripwire() -> None:
    """Guardrail passes when LLM's is_consistent=True matches recomputed=True.

    Nutrition: 10g protein, 2g fat, 10g carbs → Atwater = 4*10+9*2+4*10 = 98 kcal.
    Reported = 100 kcal → rel_pct = 2% < 15% → consistent=True.
    """
    n = _nutrition(kcal=100.0, protein=10.0, fat=2.0, carbs=10.0)
    atwater = make_atwater_check(n)
    assert atwater.is_consistent  # sanity: factory should agree
    result = await _run(n, atwater)
    assert result.output.tripwire_triggered is False


async def test_consistently_inconsistent_no_tripwire() -> None:
    """Guardrail passes when LLM correctly reports is_consistent=False."""
    # Macros → 125 kcal expected; reported = 500 kcal → rel=75% → inconsistent
    n = _nutrition(kcal=500.0, protein=10.0, fat=5.0, carbs=10.0)
    atwater = AtwaterCheck(
        nutrition=n,
        expected_kcal=125.0,
        reported_kcal=500.0,
        absolute_delta=375.0,
        relative_delta_pct=75.0,
        is_consistent=False,
    )
    result = await _run(n, atwater)
    assert result.output.tripwire_triggered is False


# ---------------------------------------------------------------------------
# Tripwire cases
# ---------------------------------------------------------------------------


async def test_hallucinated_consistent_trips_tripwire() -> None:
    """Guardrail trips when LLM reports is_consistent=True but macros say False.

    Simulates kJ-vs-kcal confusion: macros ~125 kcal but LLM claims 500 is consistent.
    """
    n = _nutrition(kcal=500.0, protein=10.0, fat=5.0, carbs=10.0)
    atwater = AtwaterCheck(
        nutrition=n,
        expected_kcal=125.0,
        reported_kcal=500.0,
        absolute_delta=375.0,
        relative_delta_pct=75.0,
        is_consistent=True,  # wrong — LLM hallucinated this
    )
    result = await _run(n, atwater)
    assert result.output.tripwire_triggered is True


async def test_hallucinated_inconsistent_trips_tripwire() -> None:
    """Guardrail trips when LLM reports is_consistent=False but macros say True."""
    # rel_pct ≈ 2% → recomputed consistent=True
    n = _nutrition(kcal=100.0, protein=10.0, fat=2.0, carbs=10.0)
    atwater = AtwaterCheck(
        nutrition=n,
        expected_kcal=98.0,
        reported_kcal=100.0,
        absolute_delta=2.0,
        relative_delta_pct=2.0,
        is_consistent=False,  # wrong — LLM hallucinated this
    )
    result = await _run(n, atwater)
    assert result.output.tripwire_triggered is True


# ---------------------------------------------------------------------------
# output_info shape
# ---------------------------------------------------------------------------


async def test_output_info_contains_debug_fields() -> None:
    """output_info always contains the four expected debug keys."""
    n = _nutrition()
    atwater = make_atwater_check(n)
    result = await _run(n, atwater)
    info = result.output.output_info
    assert "recomputed_is_consistent" in info
    assert "agent_is_consistent" in info
    assert "mismatch" in info
    assert "rel_pct" in info


async def test_boundary_exactly_at_15pct_is_consistent() -> None:
    """rel_pct just below 15% threshold is still consistent."""
    # carbs=20g only → expected = 4*20 = 80 kcal; reported = 94 kcal
    # rel = |94-80|/94 ≈ 14.9% → consistent
    n = NutritionPer100g(
        calories_kcal=94.0,
        protein_g=0.0,
        fat_g=0.0,
        saturated_fat_g=0.0,
        carbohydrates_g=20.0,
        sugar_g=5.0,
        fiber_g=1.0,
        sodium_mg=0.0,
    )
    atwater = make_atwater_check(n)
    assert atwater.is_consistent  # sanity
    result = await _run(n, atwater)
    assert result.output.tripwire_triggered is False
