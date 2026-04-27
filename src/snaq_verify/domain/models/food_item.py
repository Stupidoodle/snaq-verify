"""Input food-item models — schema 1:1 with food_items.json."""

from pydantic import BaseModel, Field


class NutritionPer100g(BaseModel):
    """Per-100g nutrition payload.

    All values are non-negative; calories are bounded above to catch obvious
    unit-confusion errors (kJ leaking in as kcal would be flagged).
    """

    calories_kcal: float = Field(..., ge=0, le=900)
    protein_g: float = Field(..., ge=0)
    fat_g: float = Field(..., ge=0)
    saturated_fat_g: float = Field(..., ge=0)
    carbohydrates_g: float = Field(..., ge=0)
    sugar_g: float = Field(..., ge=0)
    fiber_g: float = Field(..., ge=0)
    sodium_mg: float = Field(..., ge=0)


class DefaultPortion(BaseModel):
    """A typical serving description for the item."""

    amount: float = Field(..., gt=0)
    unit: str
    description: str


class FoodItem(BaseModel):
    """A single food item from the input file."""

    id: str
    name: str
    brand: str | None = None
    category: str | None = None
    barcode: str | None = None
    default_portion: DefaultPortion
    nutrition_per_100g: NutritionPer100g
