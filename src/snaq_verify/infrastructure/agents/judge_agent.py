"""Judge agent factory — builds the OpenAI Agents SDK Agent for eval scoring."""

from __future__ import annotations

from agents import Agent

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.eval_models import JudgeVerdict

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_JUDGE_INSTRUCTIONS = """\
You are an expert evaluator for a nutrition-data verification system.

You will receive:
  - A verification result produced by an automated verifier (ItemVerification JSON).
  - The ground-truth nutrition for the same food item (GroundTruthEntry JSON).

Your task is to score the verification on a 0.0–1.0 scale and determine whether
the verifier reached the correct verdict.

SCORING RUBRIC:
  1.0  — Perfect: verdict correct, all key nutrients within ±10 % of ground truth,
          proposed_correction (if present) matches ground truth closely.
  0.7–0.9 — Good: verdict is correct but 1–2 nutrients are slightly off (10–20 %).
  0.4–0.6 — Partial: verdict direction is right (e.g., flagged a discrepancy) but
             the magnitude assessment or affected nutrients differ materially.
  0.1–0.3 — Poor: verdict is wrong OR major discrepancy was missed.
  0.0  — Fail: completely wrong verdict and fabricated or missing evidence.

CORRECT VERDICT DEFINITION:
  - Return correct_verdict=True if and only if the verifier's item-level verdict
    (match / minor_discrepancy / major_discrepancy / no_data / low_confidence)
    aligns with what the ground-truth data implies:
      * If ground-truth nutrition differs from the reported nutrition by >15 % on
        any key macro (kcal, protein, fat, carbs) → expected verdict is
        major_discrepancy.
      * 5–15 % difference on any key macro → minor_discrepancy.
      * <5 % on all key macros → match.

Provide a concise reasoning string (2–4 sentences) explaining your score.
Do not fabricate numeric values — base your reasoning solely on the data provided.
"""

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_judge_agent(settings: Settings) -> Agent[None]:
    """Build the judge Agent with no tools and structured output.

    The judge is a pure structured-output call — no tools.  It receives a
    single message containing both the ``ItemVerification`` and the
    ``GroundTruthEntry``, and returns a ``JudgeVerdict``.

    Args:
        settings: Application settings providing the model pin.

    Returns:
        Configured ``Agent[None]`` with ``output_type=JudgeVerdict``.
    """
    return Agent(
        name="judge",
        instructions=_JUDGE_INSTRUCTIONS,
        tools=[],
        output_type=JudgeVerdict,
        model=settings.OPENAI_MODEL,
    )
