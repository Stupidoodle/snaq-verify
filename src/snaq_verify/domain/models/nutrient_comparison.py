"""Nutrient-level comparison models."""

from pydantic import BaseModel, Field

from snaq_verify.domain.models.enums import Verdict


class NutrientDelta(BaseModel):
    """Per-nutrient comparison: what was reported vs what we observed."""

    nutrient: str  # field name, e.g. "calories_kcal", "protein_g"
    reported: float
    observed: float
    absolute_delta: float
    relative_delta_pct: float | None = None  # None if observed below absolute floor


class NutrientVerdict(BaseModel):
    """A delta plus its categorical verdict."""

    nutrient: str
    delta: NutrientDelta
    verdict: Verdict


class ItemVerdictBundle(BaseModel):
    """All per-nutrient verdicts for one comparison plus the rolled-up verdict.

    The item verdict is the worst case across nutrients (major dominates minor
    dominates match). NO_DATA and LOW_CONFIDENCE are decided upstream and
    override.
    """

    per_nutrient: list[NutrientVerdict] = Field(default_factory=list)
    item_verdict: Verdict
