"""Assign categorical verdicts to nutrient deltas and roll up an item verdict."""

from agents import function_tool

from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.nutrient_comparison import (
    ItemVerdictBundle,
    NutrientDelta,
    NutrientVerdict,
)

# Severity ordering: higher value = worse.
_SEVERITY: dict[Verdict, int] = {
    Verdict.MATCH: 0,
    Verdict.MINOR_DISCREPANCY: 1,
    Verdict.MAJOR_DISCREPANCY: 2,
}


def _per_nutrient_verdict(
    delta: NutrientDelta,
    match_tolerance_pct: float,
    minor_tolerance_pct: float,
    absolute_floor_g: float,
) -> Verdict:
    """Classify a single :class:`NutrientDelta` into a :class:`Verdict`."""
    if delta.relative_delta_pct is not None:
        magnitude = abs(delta.relative_delta_pct)
        if magnitude <= match_tolerance_pct:
            return Verdict.MATCH
        if magnitude <= minor_tolerance_pct:
            return Verdict.MINOR_DISCREPANCY
        return Verdict.MAJOR_DISCREPANCY
    # Absolute fallback (relative suppressed by floor).
    if delta.absolute_delta <= absolute_floor_g:
        return Verdict.MATCH
    return Verdict.MINOR_DISCREPANCY


def verdict_from_deltas(
    deltas: list[NutrientDelta],
    match_tolerance_pct: float,
    minor_tolerance_pct: float,
    absolute_floor_g: float,
) -> ItemVerdictBundle:
    """Assign per-nutrient verdicts and roll up the worst-case item verdict.

    Verdict assignment per nutrient:

    - When ``relative_delta_pct`` is available:
      - ``|rel_delta| ≤ match_tolerance_pct`` → ``match``
      - ``|rel_delta| ≤ minor_tolerance_pct`` → ``minor_discrepancy``
      - otherwise → ``major_discrepancy``
    - When ``relative_delta_pct is None`` (floor suppression):
      - ``absolute_delta ≤ absolute_floor_g`` → ``match``
      - otherwise → ``minor_discrepancy``

    The item verdict is the worst verdict across all nutrients
    (``major_discrepancy`` > ``minor_discrepancy`` > ``match``).  An empty
    *deltas* list returns ``match``.

    Args:
        deltas: Per-nutrient deltas from :func:`compute_per_nutrient_delta`.
        match_tolerance_pct: |relative delta| threshold for a match (%).
        minor_tolerance_pct: |relative delta| upper bound for minor (%).
        absolute_floor_g: Absolute delta threshold used when relative is None.

    Returns:
        :class:`ItemVerdictBundle` with per-nutrient verdicts and item verdict.
    """
    per_nutrient: list[NutrientVerdict] = []
    worst: Verdict = Verdict.MATCH

    for delta in deltas:
        v = _per_nutrient_verdict(delta, match_tolerance_pct, minor_tolerance_pct, absolute_floor_g)
        per_nutrient.append(NutrientVerdict(nutrient=delta.nutrient, delta=delta, verdict=v))
        if _SEVERITY[v] > _SEVERITY[worst]:
            worst = v

    return ItemVerdictBundle(per_nutrient=per_nutrient, item_verdict=worst)


verdict_from_deltas_tool = function_tool(verdict_from_deltas)
