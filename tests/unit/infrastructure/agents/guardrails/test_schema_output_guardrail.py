"""Unit tests for schema_output_guardrail."""

from agents import RunContextWrapper, set_tracing_disabled

from snaq_verify.domain.models.enums import ConfidenceLevel, Verdict
from snaq_verify.domain.models.item_verification import ItemVerification, SourceEvidence
from snaq_verify.domain.models.nutrient_comparison import ItemVerdictBundle
from snaq_verify.infrastructure.agents.guardrails.schema_output_guardrail import (
    schema_output_guardrail,
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
# Happy path — tripwire stays off
# ---------------------------------------------------------------------------


async def test_valid_verification_no_tripwire() -> None:
    """A well-formed verification (non-empty evidence, non-blank summary) passes."""
    v = make_item_verification()
    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert result.output.tripwire_triggered is False


async def test_multiple_evidence_entries_pass() -> None:
    """Two evidence sources both valid → no tripwire."""
    item = make_food_item()
    nutrition = item.nutrition_per_100g
    candidate_usda = make_selected_candidate(source="usda", nutrition=nutrition)
    candidate_off = make_selected_candidate(source="off", nutrition=nutrition)
    bundle = ItemVerdictBundle(per_nutrient=[], item_verdict=Verdict.MATCH)

    v = make_item_verification(item=item)
    v = v.model_copy(
        update={
            "evidence": [
                SourceEvidence(source="usda", candidate=candidate_usda, bundle=bundle),
                SourceEvidence(source="off", candidate=candidate_off, bundle=bundle),
            ],
            "summary": (
                "Banana: cross-checked against 2 source(s). "
                "Overall verdict: match. 0 nutrient(s) flagged."
            ),
        },
    )

    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert result.output.tripwire_triggered is False


# ---------------------------------------------------------------------------
# Tripwire — empty evidence
# ---------------------------------------------------------------------------


async def test_empty_evidence_trips_tripwire() -> None:
    """An empty evidence list triggers the tripwire."""
    item = make_food_item()
    nutrition = item.nutrition_per_100g
    v = ItemVerification(
        item_id=item.id,
        item_name=item.name,
        reported_nutrition=nutrition,
        verdict=Verdict.NO_DATA,
        confidence=ConfidenceLevel.LOW,
        evidence=[],  # ← the violation
        proposed_correction=None,
        atwater_check_input=make_atwater_check(nutrition),
        summary="No data found.",
        notes=[],
    )

    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert result.output.tripwire_triggered is True
    issues = result.output.output_info["schema_issues"]
    assert any("evidence" in msg for msg in issues)


# ---------------------------------------------------------------------------
# Tripwire — blank summary
# ---------------------------------------------------------------------------


async def test_blank_summary_trips_tripwire() -> None:
    """A blank/whitespace summary triggers the tripwire."""
    v = make_item_verification()
    v = v.model_copy(update={"summary": "   "})

    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert result.output.tripwire_triggered is True
    issues = result.output.output_info["schema_issues"]
    assert any("summary" in msg for msg in issues)


async def test_empty_string_summary_trips_tripwire() -> None:
    """An empty-string summary triggers the tripwire."""
    v = make_item_verification()
    v = v.model_copy(update={"summary": ""})

    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert result.output.tripwire_triggered is True


# ---------------------------------------------------------------------------
# output_info shape
# ---------------------------------------------------------------------------


async def test_output_info_always_has_schema_issues_key() -> None:
    """output_info.schema_issues is always present (empty list on success)."""
    v = make_item_verification()
    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert "schema_issues" in result.output.output_info
    assert result.output.output_info["schema_issues"] == []


async def test_multiple_violations_all_reported() -> None:
    """Both an empty evidence list and a blank summary are reported in issues."""
    item = make_food_item()
    nutrition = item.nutrition_per_100g
    v = ItemVerification(
        item_id=item.id,
        item_name=item.name,
        reported_nutrition=nutrition,
        verdict=Verdict.NO_DATA,
        confidence=ConfidenceLevel.LOW,
        evidence=[],
        proposed_correction=None,
        atwater_check_input=make_atwater_check(nutrition),
        summary="",
        notes=[],
    )

    result = await schema_output_guardrail.run(context=_CTX, agent=None, agent_output=v)  # type: ignore[arg-type]
    assert result.output.tripwire_triggered is True
    issues = result.output.output_info["schema_issues"]
    assert len(issues) >= 2
