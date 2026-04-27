"""Source lookup models — what each external source returns."""

from pydantic import BaseModel, Field

from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.food_item import NutritionPer100g


class SourceQuery(BaseModel):
    """Generalized lookup query passed to a NutritionSource adapter.

    Adapters interpret the fields differently — USDA uses `name` (+ optional
    `brand`); OFF prefers `barcode` first, falls back to `name + brand`; web
    search composes a free-form query.
    """

    name: str
    brand: str | None = None
    barcode: str | None = None
    category: str | None = None


class USDACandidate(BaseModel):
    """One USDA FoodData Central search hit, possibly hydrated with details."""

    fdc_id: int
    description: str
    data_type: USDADataType
    brand_owner: str | None = None
    nutrition_per_100g: NutritionPer100g | None = None
    relevance_score: float | None = None  # USDA's own search score


class OFFProduct(BaseModel):
    """One Open Food Facts product."""

    code: str  # barcode (EAN/UPC)
    product_name: str | None = None
    brands: str | None = None
    nutrition_per_100g: NutritionPer100g | None = None
    completeness: float | None = None
    popularity_key: int | None = None


class WebSnippet(BaseModel):
    """One web search result (e.g., from Tavily)."""

    url: str
    title: str
    content: str
    score: float | None = None  # Tavily's relevance


class SelectedCandidate(BaseModel):
    """A single best-match candidate selected by `select_best_candidate`.

    Normalized to a uniform shape so downstream tools don't care which source
    produced it.
    """

    source: str = Field(..., pattern="^(usda|off|web)$")
    source_id: str  # fdcId / barcode / url
    source_name: str  # human-readable label
    nutrition_per_100g: NutritionPer100g
    match_score: float = Field(..., ge=0, le=1)
    notes: str | None = None
