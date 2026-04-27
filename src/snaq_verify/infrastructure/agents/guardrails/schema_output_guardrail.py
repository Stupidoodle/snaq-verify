"""Schema output guardrail — verifies ItemVerification structural invariants."""

from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, output_guardrail

from snaq_verify.domain.models.item_verification import ItemVerification


@output_guardrail
async def schema_output_guardrail(
    ctx: RunContextWrapper[Any],
    agent: Agent[Any],
    output: ItemVerification,
) -> GuardrailFunctionOutput:
    """Enforce structural invariants on the agent's ``ItemVerification`` output.

    Pydantic validates individual field types, but cannot enforce cross-field
    business rules.  This guardrail catches:

    - **Empty evidence**: the agent must have queried at least one external
      source.  An empty evidence list means the numeric verdict was fabricated.
    - **Missing summary**: the agent must have called ``format_human_summary``.
    - **Match score out of range**: every evidence candidate's ``match_score``
      must be in [0, 1] (Pydantic validates this too, but we double-check).

    Args:
        ctx: Run context (not used directly; provided by the SDK).
        agent: The agent that produced the output.
        output: The ``ItemVerification`` to validate.

    Returns:
        ``GuardrailFunctionOutput`` with ``tripwire_triggered=True`` when any
        invariant is violated.  ``output_info`` carries a list of issue strings
        for debugging.
    """
    issues: list[str] = []

    if not output.evidence:
        issues.append(
            "evidence list is empty — "
            "the agent must query at least one external source",
        )

    if not output.summary.strip():
        issues.append("summary is blank — format_human_summary must be called")

    for ev in output.evidence:
        score = ev.candidate.match_score
        if not (0.0 <= score <= 1.0):
            issues.append(
                f"candidate {ev.candidate.source_id!r} has out-of-range "
                f"match_score={score}",
            )

    return GuardrailFunctionOutput(
        output_info={"schema_issues": issues},
        tripwire_triggered=bool(issues),
    )
