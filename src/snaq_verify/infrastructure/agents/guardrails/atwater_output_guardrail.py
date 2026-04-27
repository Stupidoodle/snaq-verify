"""Atwater output guardrail — verifies the agent's Atwater check is correct."""

from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, output_guardrail

from snaq_verify.domain.models.item_verification import ItemVerification

# Atwater factors (kcal / gram)
_PROTEIN_FACTOR: float = 4.0
_CARB_FACTOR: float = 4.0
_FAT_FACTOR: float = 9.0

# Must match Settings.ATWATER_TOLERANCE_PCT — hardcoded here to avoid a
# circular dependency at import time (settings is not always available when
# the agent module is imported).
_TOLERANCE_PCT: float = 15.0


@output_guardrail
async def atwater_output_guardrail(
    ctx: RunContextWrapper[Any],
    agent: Agent[Any],
    output: ItemVerification,
) -> GuardrailFunctionOutput:
    """Re-compute the Atwater check and compare it to the agent's answer.

    The agent calls ``check_atwater_consistency`` as a tool and embeds the
    result in ``atwater_check_input``.  This guardrail independently
    re-derives ``is_consistent`` from the ``reported_nutrition`` macros and
    verifies the flag matches what the LLM reported.  A mismatch means the
    LLM hallucinated part of the check, so the tripwire fires.

    Args:
        ctx: Run context (not used directly; provided by the SDK).
        agent: The agent that produced the output.
        output: The ``ItemVerification`` produced by the verifier agent.

    Returns:
        ``GuardrailFunctionOutput`` with ``tripwire_triggered=True`` when the
        agent's ``is_consistent`` flag does not agree with the recomputed value.
    """
    n = output.reported_nutrition
    expected_kcal = (
        _PROTEIN_FACTOR * n.protein_g
        + _CARB_FACTOR * n.carbohydrates_g
        + _FAT_FACTOR * n.fat_g
    )
    abs_delta = abs(n.calories_kcal - expected_kcal)
    rel_pct = abs_delta / max(n.calories_kcal, 1.0) * 100.0
    expected_consistent = rel_pct <= _TOLERANCE_PCT

    agent_consistent = output.atwater_check_input.is_consistent
    mismatch = agent_consistent != expected_consistent

    return GuardrailFunctionOutput(
        output_info={
            "recomputed_is_consistent": expected_consistent,
            "agent_is_consistent": agent_consistent,
            "rel_pct": round(rel_pct, 2),
            "mismatch": mismatch,
        },
        tripwire_triggered=mismatch,
    )
