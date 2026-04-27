"""Format a human-readable summary for a food-item verification result."""

from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.nutrient_comparison import ItemVerdictBundle


def format_human_summary(
    item: FoodItem,
    verdict_bundle: ItemVerdictBundle,
    evidence_count: int,
) -> str:
    """Produce a one-paragraph plain-text summary of a verification result.

    This is a **pure string template** — it never calls an LLM.  Calling it
    twice with identical arguments always returns the same string.

    The summary includes:
    - Item name (and brand in parentheses if present).
    - Number of sources cross-checked.
    - Overall verdict (``match`` / ``minor_discrepancy`` / ``major_discrepancy``
      or any other :class:`~snaq_verify.domain.models.enums.Verdict` value).
    - Number of nutrients flagged (verdict ≠ ``match``).

    Args:
        item: The food item being verified.
        verdict_bundle: The verdict bundle produced by
            :func:`verdict_from_deltas`.
        evidence_count: How many external sources were consulted.

    Returns:
        A short human-readable paragraph (one sentence per piece of evidence).
    """
    brand_part = f" ({item.brand})" if item.brand else ""
    verdict_label = verdict_bundle.item_verdict.value  # e.g. "match", "minor_discrepancy"
    flagged = sum(
        1
        for nv in verdict_bundle.per_nutrient
        if nv.verdict != Verdict.MATCH
    )

    return (
        f"{item.name}{brand_part}: cross-checked against {evidence_count} source(s). "
        f"Overall verdict: {verdict_label}. "
        f"{flagged} nutrient(s) flagged."
    )
