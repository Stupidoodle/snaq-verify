"""Confidence output guardrail — enforces deterministic confidence derivation."""

from __future__ import annotations

from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, output_guardrail

from snaq_verify.domain.models.enums import ConfidenceLevel, Verdict
from snaq_verify.domain.models.item_verification import ItemVerification

# ---------------------------------------------------------------------------
# Thresholds (match team-lead spec)
# ---------------------------------------------------------------------------

_HIGH_SCORE_THRESHOLD = 0.85
_MEDIUM_SCORE_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Deterministic derivation (also imported by the adapter for post-processing)
# ---------------------------------------------------------------------------


def derive_confidence(output: ItemVerification) -> ConfidenceLevel:
    """Deterministically derive confidence from evidence in an ItemVerification.

    Rules applied in priority order:

    - **HIGH**   : ≥2 evidence sources, top candidate ``match_score`` ≥ 0.85,
                   and item verdict is ``MATCH`` or ``MINOR_DISCREPANCY`` (sources
                   broadly agree with the reported nutrition).
    - **MEDIUM** : top candidate ``match_score`` ≥ 0.70, OR ≥2 sources used
                   even if they disagree on magnitude.
    - **LOW**    : everything else (0–1 source with a low score, no data, etc.).

    Args:
        output: The ``ItemVerification`` to evaluate.

    Returns:
        The deterministically derived ``ConfidenceLevel``.
    """
    evidence = output.evidence
    n_sources = len(evidence)

    if n_sources == 0:
        return ConfidenceLevel.LOW

    max_score = max(e.candidate.match_score for e in evidence)
    verdict_allows_high = output.verdict in (
        Verdict.MATCH,
        Verdict.MINOR_DISCREPANCY,
    )

    if n_sources >= 2 and max_score >= _HIGH_SCORE_THRESHOLD and verdict_allows_high:
        return ConfidenceLevel.HIGH

    if max_score >= _MEDIUM_SCORE_THRESHOLD or n_sources >= 2:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


@output_guardrail
async def confidence_output_guardrail(
    ctx: RunContextWrapper[Any],
    agent: Agent[Any],
    output: ItemVerification,
) -> GuardrailFunctionOutput:
    """Verify that the LLM's confidence choice matches the deterministic rule.

    The verifier's instructions spell out the confidence rule verbatim.  This
    guardrail re-derives confidence from the evidence and trips if the LLM's
    value differs — giving it a chance to self-correct on retry.  The adapter
    also overrides confidence post-run, so this acts as an in-band audit.

    ``output_info`` always carries:

    - ``derived_confidence``  — what the rule dictates
    - ``agent_confidence``    — what the LLM reported
    - ``n_sources``           — number of evidence entries used
    - ``max_match_score``     — best candidate score in evidence
    - ``mismatch``            — True when the guardrail trips

    Args:
        ctx: Run context (not used directly).
        agent: The agent that produced the output.
        output: The ``ItemVerification`` to validate.

    Returns:
        ``GuardrailFunctionOutput`` with ``tripwire_triggered=True`` when the
        LLM's confidence does not match the deterministic derivation.
    """
    derived = derive_confidence(output)
    n_sources = len(output.evidence)
    max_score = max(
        (e.candidate.match_score for e in output.evidence), default=0.0,
    )
    mismatch = output.confidence != derived

    return GuardrailFunctionOutput(
        output_info={
            "derived_confidence": derived.value,
            "agent_confidence": output.confidence.value,
            "n_sources": n_sources,
            "max_match_score": round(max_score, 4),
            "mismatch": mismatch,
        },
        tripwire_triggered=mismatch,
    )
