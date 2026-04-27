"""Enums shared across the domain."""

from enum import Enum


class USDADataType(str, Enum):
    """USDA FoodData Central data-type filters.

    Priority for generic raw foods: Foundation > SR Legacy > Branded > Survey.
    """

    FOUNDATION = "Foundation"
    SR_LEGACY = "SR Legacy"
    BRANDED = "Branded"
    SURVEY_FNDDS = "Survey (FNDDS)"


class Verdict(str, Enum):
    """Verdict for a single nutrient or for an item as a whole.

    Order from best to worst: MATCH < MINOR_DISCREPANCY < MAJOR_DISCREPANCY.
    NO_DATA and LOW_CONFIDENCE are orthogonal — they signal absence or
    insufficient evidence rather than agreement/disagreement.
    """

    MATCH = "match"
    MINOR_DISCREPANCY = "minor_discrepancy"
    MAJOR_DISCREPANCY = "major_discrepancy"
    NO_DATA = "no_data"
    LOW_CONFIDENCE = "low_confidence"


class ConfidenceLevel(str, Enum):
    """How much trust the system places in a per-item verdict."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
