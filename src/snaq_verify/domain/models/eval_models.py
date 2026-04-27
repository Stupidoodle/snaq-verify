"""Eval-layer models — golden set entries and judge verdicts."""

from datetime import datetime

from pydantic import BaseModel, Field

from snaq_verify.domain.models.food_item import NutritionPer100g


class GroundTruthEntry(BaseModel):
    """Hand-curated authoritative nutrition for one food item.

    Each entry must cite an authoritative source URL so a reviewer can audit
    the golden set. Variability disclaimers (farmed-vs-wild, brand variance)
    live in `notes`.
    """

    item_id: str
    item_name: str
    source: str
    source_url: str
    nutrition_per_100g: NutritionPer100g
    notes: str | None = None


class JudgeVerdict(BaseModel):
    """LLM judge output for one item — the judge agent's `output_type`."""

    item_id: str
    score: float = Field(..., ge=0, le=1)
    correct_verdict: bool
    reasoning: str


class EvalRunMetadata(BaseModel):
    """Provenance for the eval run."""

    timestamp: datetime
    model: str
    item_count: int


class EvalReport(BaseModel):
    """The bonus deliverable written to `eval_report.json`."""

    metadata: EvalRunMetadata
    judgments: list[JudgeVerdict] = Field(default_factory=list)
    aggregate_score: float = Field(..., ge=0, le=1)
    correct_verdicts: int
    total: int
