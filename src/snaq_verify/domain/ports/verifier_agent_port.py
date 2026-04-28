"""Verifier agent port — orchestrates tool calls to verify one food item."""

from __future__ import annotations

from abc import ABC, abstractmethod

from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification


class VerifierAgentPort(ABC):
    """Abstract interface for the per-item verifier agent.

    The adapter wraps an OpenAI Agents SDK ``Agent`` whose ``output_type`` is
    ``ItemVerification``.  The agent's tools are deterministic functions that
    return structured data — the LLM cannot fabricate numeric fields.
    """

    @abstractmethod
    async def verify(
        self, item: FoodItem, hint: str | None = None,
    ) -> ItemVerification:
        """Verify a single food item.

        Args:
            item: The input food item with reported ``nutrition_per_100g``.
            hint: Optional feedback from a prior judge run (e.g. ``"evidence
                is thin — try SR Legacy as a fallback source"``).  When
                provided the hint is prepended to the agent prompt so the
                agent can self-correct without repeating all tool calls.
                Defaults to ``None`` (standard verification with no extra
                context).

        Returns:
            A structured verification result.  Output guardrails ensure the
            verdict is consistent with the synthesized nutrition (Atwater
            check) and the schema is valid.
        """
        raise NotImplementedError
