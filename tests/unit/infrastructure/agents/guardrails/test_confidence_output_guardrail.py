"""Unit tests for confidence_output_guardrail and derive_confidence."""

from agents import RunContextWrapper, set_tracing_disabled

from snaq_verify.domain.models.enums import ConfidenceLevel, Verdict
from snaq_verify.domain.models.item_verification import ItemVerification, SourceEvidence
from snaq_verify.domain.models.nutrient_comparison import ItemVerdictBundle
from snaq_verify.infrastructure.agents.guardrails.confidence_output_guardrail import (
    confidence_output_guardrail,
    derive_confidence,
)
from tests.fakes.factories import (
    make_atwater_check,
    make_food_item,
    make_item_verification,
    make_selected_candidate,
)

set_tracing_disabled(True)

_CTX = RunContextWrapper(context=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verification(
    *,
    n_sources: int = 1,
    match_score: float = 0.95,
    verdict: Verdict = Verdict.MATCH,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> ItemVerification:
    """Build an ItemVerification with controlled evidence shape."""
    item = make_food_item()
    nutrition = item.nutrition_per_100g
    bundle = ItemVerdictBundle(per_nutrient=[], item_verdict=verdict)

    sources = ["usda", "off", "web"]
    evidence = [
        SourceEvidence(
            source=sources[i % len(sources)],
            candidate=make_selected_candidate(
                source=sources[i % len(sources)],
                nutrition=nutrition,
            ).model_copy(update={"match_score": match_score}),
            bundle=bundle,
        )
        for i in range(n_sources)
    ]

    v = make_item_verification(item=item, verdict=verdict, confidence=confidence)
    return v.model_copy(update={"evidence": evidence, "verdict": verdict})


async def _run(v: ItemVerification):  # type: ignore[no-untyped-def]
    return await confidence_output_guardrail.run(
        context=_CTX, agent=None, agent_output=v,
    )


# ---------------------------------------------------------------------------
# derive_confidence — unit tests
# ---------------------------------------------------------------------------


def test_derive_confidence_no_evidence_returns_low() -> None:
    """Zero evidence sources always yields LOW."""
    v = make_item_verification()
    v = v.model_copy(update={"evidence": []})
    assert derive_confidence(v) == ConfidenceLevel.LOW


def test_derive_confidence_two_sources_high_score_match_returns_high() -> None:
    """≥2 sources + score ≥ 0.85 + MATCH verdict → HIGH."""
    v = _make_verification(n_sources=2, match_score=0.90, verdict=Verdict.MATCH)
    assert derive_confidence(v) == ConfidenceLevel.HIGH


def test_derive_confidence_two_sources_high_score_minor_returns_high() -> None:
    """≥2 sources + score ≥ 0.85 + MINOR_DISCREPANCY verdict → HIGH."""
    v = _make_verification(
        n_sources=2, match_score=0.85, verdict=Verdict.MINOR_DISCREPANCY,
    )
    assert derive_confidence(v) == ConfidenceLevel.HIGH


def test_derive_confidence_two_sources_high_score_major_returns_medium() -> None:
    """≥2 sources + score ≥ 0.85, but MAJOR_DISCREPANCY verdict → MEDIUM (not HIGH)."""
    v = _make_verification(
        n_sources=2, match_score=0.90, verdict=Verdict.MAJOR_DISCREPANCY,
    )
    assert derive_confidence(v) == ConfidenceLevel.MEDIUM


def test_derive_confidence_one_source_high_score_returns_medium() -> None:
    """1 source with score ≥ 0.70 → MEDIUM (not HIGH — need ≥2 for HIGH)."""
    v = _make_verification(n_sources=1, match_score=0.95, verdict=Verdict.MATCH)
    assert derive_confidence(v) == ConfidenceLevel.MEDIUM


def test_derive_confidence_two_sources_low_score_returns_medium() -> None:
    """≥2 sources but score < 0.85 → MEDIUM (n_sources ≥ 2 alone qualifies)."""
    v = _make_verification(n_sources=2, match_score=0.60, verdict=Verdict.MATCH)
    assert derive_confidence(v) == ConfidenceLevel.MEDIUM


def test_derive_confidence_one_source_below_medium_threshold_returns_low() -> None:
    """1 source with score < 0.70 → LOW."""
    v = _make_verification(n_sources=1, match_score=0.50, verdict=Verdict.MATCH)
    assert derive_confidence(v) == ConfidenceLevel.LOW


def test_derive_confidence_boundary_exactly_085_is_high_eligible() -> None:
    """Score exactly at 0.85 with 2 sources + MATCH → HIGH."""
    v = _make_verification(n_sources=2, match_score=0.85, verdict=Verdict.MATCH)
    assert derive_confidence(v) == ConfidenceLevel.HIGH


def test_derive_confidence_boundary_exactly_070_is_medium_eligible() -> None:
    """Score exactly at 0.70 → MEDIUM (≥ threshold)."""
    v = _make_verification(n_sources=1, match_score=0.70, verdict=Verdict.MATCH)
    assert derive_confidence(v) == ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# confidence_output_guardrail — happy path (no tripwire)
# ---------------------------------------------------------------------------


async def test_correct_high_confidence_no_tripwire() -> None:
    """Guardrail passes when agent correctly reports HIGH (2 sources, score ≥0.85)."""
    v = _make_verification(
        n_sources=2,
        match_score=0.90,
        verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.HIGH,
    )
    result = await _run(v)
    assert result.output.tripwire_triggered is False


async def test_correct_medium_confidence_no_tripwire() -> None:
    """Guardrail passes when agent correctly reports MEDIUM (1 source, score ≥0.70)."""
    v = _make_verification(
        n_sources=1,
        match_score=0.80,
        verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.MEDIUM,
    )
    result = await _run(v)
    assert result.output.tripwire_triggered is False


async def test_correct_low_confidence_no_tripwire() -> None:
    """Guardrail passes when agent correctly reports LOW (1 source, score <0.70)."""
    v = _make_verification(
        n_sources=1,
        match_score=0.55,
        verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.LOW,
    )
    result = await _run(v)
    assert result.output.tripwire_triggered is False


# ---------------------------------------------------------------------------
# confidence_output_guardrail — tripwire cases
# ---------------------------------------------------------------------------


async def test_low_when_high_expected_trips_tripwire() -> None:
    """Agent reports LOW but evidence warrants HIGH → tripwire fires."""
    v = _make_verification(
        n_sources=2,
        match_score=0.90,
        verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.LOW,  # wrong
    )
    result = await _run(v)
    assert result.output.tripwire_triggered is True
    assert result.output.output_info["derived_confidence"] == "high"
    assert result.output.output_info["agent_confidence"] == "low"


async def test_high_when_medium_expected_trips_tripwire() -> None:
    """Agent reports HIGH but only 1 source → should be MEDIUM → tripwire fires."""
    v = _make_verification(
        n_sources=1,
        match_score=0.95,
        verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.HIGH,  # wrong — 1 source can't be HIGH
    )
    result = await _run(v)
    assert result.output.tripwire_triggered is True
    assert result.output.output_info["derived_confidence"] == "medium"


async def test_medium_when_low_expected_trips_tripwire() -> None:
    """Agent reports MEDIUM but score is 0.5 → should be LOW → tripwire fires."""
    v = _make_verification(
        n_sources=1,
        match_score=0.50,
        verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.MEDIUM,  # wrong
    )
    result = await _run(v)
    assert result.output.tripwire_triggered is True
    assert result.output.output_info["derived_confidence"] == "low"


# ---------------------------------------------------------------------------
# output_info shape
# ---------------------------------------------------------------------------


async def test_output_info_always_has_expected_keys() -> None:
    """output_info always contains the five expected debug keys."""
    v = _make_verification(
        n_sources=2, match_score=0.90, verdict=Verdict.MATCH,
        confidence=ConfidenceLevel.HIGH,
    )
    result = await _run(v)
    info = result.output.output_info
    assert "derived_confidence" in info
    assert "agent_confidence" in info
    assert "n_sources" in info
    assert "max_match_score" in info
    assert "mismatch" in info


async def test_output_info_no_evidence_reports_zero_score() -> None:
    """With empty evidence, max_match_score=0.0 and derived=low."""
    item = make_food_item()
    n = item.nutrition_per_100g
    v = ItemVerification(
        item_id=item.id,
        item_name=item.name,
        reported_nutrition=n,
        verdict=Verdict.NO_DATA,
        confidence=ConfidenceLevel.LOW,
        evidence=[],
        proposed_correction=None,
        atwater_check_input=make_atwater_check(n),
        summary="No data.",
        notes=[],
    )
    result = await _run(v)
    assert result.output.output_info["max_match_score"] == 0.0
    assert result.output.output_info["n_sources"] == 0
    assert result.output.tripwire_triggered is False
