"""Table-driven tests for verdict_from_deltas."""

import pytest

from snaq_verify.application.tools.verdict_from_deltas import verdict_from_deltas
from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.nutrient_comparison import ItemVerdictBundle, NutrientDelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MATCH_PCT = 5.0
MINOR_PCT = 15.0
FLOOR_G = 0.5


def _delta(
    nutrient: str = "protein_g",
    reported: float = 10.0,
    observed: float = 10.0,
    relative_delta_pct: float | None = 0.0,
) -> NutrientDelta:
    return NutrientDelta(
        nutrient=nutrient,
        reported=reported,
        observed=observed,
        absolute_delta=abs(reported - observed),
        relative_delta_pct=relative_delta_pct,
    )


# ---------------------------------------------------------------------------
# Parametrized verdict-assignment cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "relative_delta_pct, absolute_delta, expected_verdict",
    [
        # Relative path: match
        (0.0, 0.0, Verdict.MATCH),
        (MATCH_PCT, 0.5, Verdict.MATCH),           # exactly at threshold
        (MATCH_PCT - 0.01, 0.5, Verdict.MATCH),    # just below
        # Relative path: minor
        (MATCH_PCT + 0.01, 1.0, Verdict.MINOR_DISCREPANCY),  # just above
        (MINOR_PCT, 2.0, Verdict.MINOR_DISCREPANCY),          # exactly at minor threshold
        (MINOR_PCT - 0.01, 2.0, Verdict.MINOR_DISCREPANCY),
        # Relative path: major
        (MINOR_PCT + 0.01, 3.0, Verdict.MAJOR_DISCREPANCY),  # just above minor
        (50.0, 5.0, Verdict.MAJOR_DISCREPANCY),
        # Absolute fallback (relative_delta_pct=None): match
        (None, FLOOR_G, Verdict.MATCH),            # exactly at floor
        (None, FLOOR_G - 0.01, Verdict.MATCH),
        (None, 0.0, Verdict.MATCH),
        # Absolute fallback: minor
        (None, FLOOR_G + 0.01, Verdict.MINOR_DISCREPANCY),
        (None, 2.0, Verdict.MINOR_DISCREPANCY),
        # Negative relative delta: magnitude matters
        (-MATCH_PCT, 0.5, Verdict.MATCH),
        (-MINOR_PCT - 0.01, 3.0, Verdict.MAJOR_DISCREPANCY),
    ],
)
def test_per_nutrient_verdict(
    relative_delta_pct: float | None,
    absolute_delta: float,
    expected_verdict: Verdict,
) -> None:
    delta = NutrientDelta(
        nutrient="protein_g",
        reported=10.0,
        observed=10.0,
        absolute_delta=absolute_delta,
        relative_delta_pct=relative_delta_pct,
    )
    bundle = verdict_from_deltas(
        [delta],
        match_tolerance_pct=MATCH_PCT,
        minor_tolerance_pct=MINOR_PCT,
        absolute_floor_g=FLOOR_G,
    )
    assert len(bundle.per_nutrient) == 1
    assert bundle.per_nutrient[0].verdict == expected_verdict


# ---------------------------------------------------------------------------
# Item-level worst-case rollup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "verdicts_in, expected_item_verdict",
    [
        ([Verdict.MATCH, Verdict.MATCH], Verdict.MATCH),
        ([Verdict.MATCH, Verdict.MINOR_DISCREPANCY], Verdict.MINOR_DISCREPANCY),
        ([Verdict.MINOR_DISCREPANCY, Verdict.MAJOR_DISCREPANCY], Verdict.MAJOR_DISCREPANCY),
        ([Verdict.MATCH, Verdict.MAJOR_DISCREPANCY, Verdict.MINOR_DISCREPANCY], Verdict.MAJOR_DISCREPANCY),
        ([Verdict.MAJOR_DISCREPANCY], Verdict.MAJOR_DISCREPANCY),
    ],
)
def test_item_verdict_worst_case(
    verdicts_in: list[Verdict],
    expected_item_verdict: Verdict,
) -> None:
    # Map verdict → relative_delta_pct that produces it
    verdict_to_pct: dict[Verdict, float] = {
        Verdict.MATCH: 1.0,
        Verdict.MINOR_DISCREPANCY: 10.0,
        Verdict.MAJOR_DISCREPANCY: 50.0,
    }
    deltas = [
        NutrientDelta(
            nutrient=f"nutrient_{i}",
            reported=10.0,
            observed=10.0,
            absolute_delta=abs(10.0 * verdict_to_pct[v] / 100),
            relative_delta_pct=verdict_to_pct[v],
        )
        for i, v in enumerate(verdicts_in)
    ]
    bundle = verdict_from_deltas(
        deltas,
        match_tolerance_pct=MATCH_PCT,
        minor_tolerance_pct=MINOR_PCT,
        absolute_floor_g=FLOOR_G,
    )
    assert bundle.item_verdict == expected_item_verdict


# ---------------------------------------------------------------------------
# Edge and degenerate cases
# ---------------------------------------------------------------------------

def test_empty_deltas_returns_match() -> None:
    bundle = verdict_from_deltas(
        [],
        match_tolerance_pct=MATCH_PCT,
        minor_tolerance_pct=MINOR_PCT,
        absolute_floor_g=FLOOR_G,
    )
    assert isinstance(bundle, ItemVerdictBundle)
    assert bundle.per_nutrient == []
    assert bundle.item_verdict == Verdict.MATCH


def test_all_nutrients_match() -> None:
    deltas = [
        _delta(nutrient=f"nutrient_{i}", relative_delta_pct=1.0)
        for i in range(8)
    ]
    bundle = verdict_from_deltas(
        deltas,
        match_tolerance_pct=MATCH_PCT,
        minor_tolerance_pct=MINOR_PCT,
        absolute_floor_g=FLOOR_G,
    )
    assert bundle.item_verdict == Verdict.MATCH
    assert all(nv.verdict == Verdict.MATCH for nv in bundle.per_nutrient)


def test_returns_item_verdict_bundle() -> None:
    bundle = verdict_from_deltas(
        [_delta()],
        match_tolerance_pct=MATCH_PCT,
        minor_tolerance_pct=MINOR_PCT,
        absolute_floor_g=FLOOR_G,
    )
    assert isinstance(bundle, ItemVerdictBundle)


def test_per_nutrient_count_matches_input() -> None:
    deltas = [_delta(nutrient=f"n_{i}") for i in range(5)]
    bundle = verdict_from_deltas(
        deltas,
        match_tolerance_pct=MATCH_PCT,
        minor_tolerance_pct=MINOR_PCT,
        absolute_floor_g=FLOOR_G,
    )
    assert len(bundle.per_nutrient) == 5
