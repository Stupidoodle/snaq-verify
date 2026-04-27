"""Factory helpers for building test fixtures without real adapters."""

from datetime import UTC, datetime

from snaq_verify.domain.models.atwater_check import AtwaterCheck
from snaq_verify.domain.models.enums import ConfidenceLevel, Verdict
from snaq_verify.domain.models.eval_models import (
    EvalReport,
    EvalRunMetadata,
    GroundTruthEntry,
    JudgeVerdict,
)
from snaq_verify.domain.models.food_item import DefaultPortion, FoodItem, NutritionPer100g
from snaq_verify.domain.models.item_verification import ItemVerification, SourceEvidence
from snaq_verify.domain.models.nutrient_comparison import ItemVerdictBundle
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.models.source_lookup import SelectedCandidate
from snaq_verify.domain.models.verification_report import RunMetadata, VerificationReport


def make_nutrition(
    *,
    calories_kcal: float = 100.0,
    protein_g: float = 10.0,
    fat_g: float = 5.0,
    saturated_fat_g: float = 1.0,
    carbohydrates_g: float = 10.0,
    sugar_g: float = 2.0,
    fiber_g: float = 1.0,
    sodium_mg: float = 50.0,
) -> NutritionPer100g:
    """Return a NutritionPer100g with sensible defaults."""
    return NutritionPer100g(
        calories_kcal=calories_kcal,
        protein_g=protein_g,
        fat_g=fat_g,
        saturated_fat_g=saturated_fat_g,
        carbohydrates_g=carbohydrates_g,
        sugar_g=sugar_g,
        fiber_g=fiber_g,
        sodium_mg=sodium_mg,
    )


def make_food_item(
    item_id: str = "test-item",
    name: str = "Test Food",
    brand: str | None = None,
    nutrition: NutritionPer100g | None = None,
) -> FoodItem:
    """Return a FoodItem with sensible defaults."""
    return FoodItem(
        id=item_id,
        name=name,
        brand=brand,
        default_portion=DefaultPortion(amount=100.0, unit="g", description="1 serving"),
        nutrition_per_100g=nutrition or make_nutrition(),
    )


def make_atwater_check(nutrition: NutritionPer100g | None = None) -> AtwaterCheck:
    """Return a consistent AtwaterCheck for the given nutrition."""
    n = nutrition or make_nutrition()
    expected = 4 * n.protein_g + 4 * n.carbohydrates_g + 9 * n.fat_g
    reported = n.calories_kcal
    abs_delta = abs(reported - expected)
    rel_pct = (abs_delta / reported * 100) if reported > 0 else 0.0
    return AtwaterCheck(
        nutrition=n,
        expected_kcal=expected,
        reported_kcal=reported,
        absolute_delta=abs_delta,
        relative_delta_pct=rel_pct,
        is_consistent=rel_pct <= 15.0,
    )


def make_selected_candidate(
    source: str = "usda",
    nutrition: NutritionPer100g | None = None,
) -> SelectedCandidate:
    """Return a SelectedCandidate with sensible defaults."""
    return SelectedCandidate(
        source=source,
        source_id="12345",
        source_name="Test Food (USDA Foundation)",
        nutrition_per_100g=nutrition or make_nutrition(),
        match_score=0.95,
    )


def make_item_verification(
    item: FoodItem | None = None,
    verdict: Verdict = Verdict.MATCH,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> ItemVerification:
    """Return an ItemVerification with sensible defaults."""
    food = item or make_food_item()
    nutrition = food.nutrition_per_100g
    candidate = make_selected_candidate(nutrition=nutrition)
    bundle = ItemVerdictBundle(per_nutrient=[], item_verdict=verdict)
    evidence = [SourceEvidence(source="usda", candidate=candidate, bundle=bundle)]
    return ItemVerification(
        item_id=food.id,
        item_name=food.name,
        reported_nutrition=nutrition,
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        proposed_correction=None,
        atwater_check_input=make_atwater_check(nutrition),
        summary=f"{food.name}: {verdict.value}",
        notes=[],
    )


def make_verification_report(
    items: list[ItemVerification] | None = None,
    model: str = "gpt-test",
) -> VerificationReport:
    """Return a VerificationReport with sensible defaults."""
    item_list = items or [make_item_verification()]
    flag_verdicts = {Verdict.MINOR_DISCREPANCY, Verdict.MAJOR_DISCREPANCY}
    flag_count = sum(1 for v in item_list if v.verdict in flag_verdicts)
    return VerificationReport(
        metadata=RunMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            input_count=len(item_list),
            flag_count=flag_count,
            model=model,
        ),
        items=item_list,
    )


def make_ground_truth_entry(
    item_id: str = "test-item",
    item_name: str = "Test Food",
    nutrition: NutritionPer100g | None = None,
) -> GroundTruthEntry:
    """Return a GroundTruthEntry with sensible defaults."""
    return GroundTruthEntry(
        item_id=item_id,
        item_name=item_name,
        source="USDA Foundation Foods",
        source_url="https://fdc.nal.usda.gov/food-details/test/nutrients",
        nutrition_per_100g=nutrition or make_nutrition(),
        notes=None,
    )


def make_judge_verdict(
    item_id: str = "test-item",
    score: float = 1.0,
    correct_verdict: bool = True,
) -> JudgeVerdict:
    """Return a JudgeVerdict with sensible defaults."""
    return JudgeVerdict(
        item_id=item_id,
        score=score,
        correct_verdict=correct_verdict,
        reasoning="Fake judge: verification matches ground truth.",
    )


def make_eval_report(
    judgments: list[JudgeVerdict] | None = None,
    model: str = "gpt-test",
) -> EvalReport:
    """Return an EvalReport with sensible defaults."""
    j_list = judgments or [make_judge_verdict()]
    agg = sum(j.score for j in j_list) / len(j_list) if j_list else 0.0
    correct = sum(1 for j in j_list if j.correct_verdict)
    return EvalReport(
        metadata=EvalRunMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            model=model,
            item_count=len(j_list),
        ),
        judgments=j_list,
        aggregate_score=agg,
        correct_verdicts=correct,
        total=len(j_list),
    )


def make_pipeline_state(**kwargs: object) -> PipelineState:
    """Return a PipelineState; keyword args override defaults."""
    return PipelineState(**kwargs)  # type: ignore[arg-type]
