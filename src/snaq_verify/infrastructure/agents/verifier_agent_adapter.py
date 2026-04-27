"""VerifierAgentAdapter — VerifierAgentPort backed by the OpenAI Agents SDK."""

from __future__ import annotations

from typing import Any

from agents import Runner

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.open_food_facts_client_port import OpenFoodFactsClientPort
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort
from snaq_verify.domain.ports.usda_client_port import USDAClientPort
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort
from snaq_verify.infrastructure.agents.guardrails.confidence_output_guardrail import (
    derive_confidence,
)
from snaq_verify.infrastructure.agents.verifier_agent import (
    VerifierContext,
    build_verifier_agent,
)


class VerifierAgentAdapter(VerifierAgentPort):
    """Implements ``VerifierAgentPort`` using an OpenAI Agents SDK ``Agent``.

    The adapter builds one ``Agent`` at construction time and reuses it for
    every ``verify`` call.  Each call creates a fresh ``VerifierContext``
    so there is no shared mutable state between concurrent verifications.

    **Post-run overrides** applied after ``Runner.run`` returns:

    - *notes*: Replaced with ``context.tool_events`` — a mechanically-derived
      list of notable IO events (404s, empty results, web-search fallbacks).
      This prevents the LLM from fabricating plausible-sounding error messages
      that did not occur.

    - *confidence*: Re-derived deterministically from evidence (number of
      sources, top match score, overall verdict) and applied even when the
      confidence guardrail does not trip.  This is the authoritative value.
    """

    def __init__(
        self,
        settings: Settings,
        logger: LoggerPort,
        usda: USDAClientPort,
        off: OpenFoodFactsClientPort,
        tavily: TavilyClientPort,
    ) -> None:
        """Construct the adapter.

        Args:
            settings: Application settings (model pin, thresholds).
            logger: Structured logger for verification events.
            usda: USDA FoodData Central client.
            off: Open Food Facts client.
            tavily: Tavily web-search client.
        """
        self._settings = settings
        self._logger = logger
        self._usda = usda
        self._off = off
        self._tavily = tavily
        self._agent = build_verifier_agent(settings)

    # ------------------------------------------------------------------
    # VerifierAgentPort
    # ------------------------------------------------------------------

    async def verify(self, item: FoodItem) -> ItemVerification:
        """Verify a single food item using the verifier agent.

        The item is serialised to JSON and passed as the agent's input message.
        The agent calls USDA / OFF / Tavily tools to gather evidence, then
        runs deterministic compute tools to produce a structured
        ``ItemVerification``.  Three output guardrails fire before the result
        is returned.

        After ``Runner.run`` completes, the adapter applies two deterministic
        overrides:

        1. ``notes`` ← ``context.tool_events`` (mechanically-derived audit
           trail, prevents LLM fabrication).
        2. ``confidence`` ← ``derive_confidence(verification)`` (exact rule
           from instructions, defeats any LLM deviation).

        Args:
            item: The food item to verify.

        Returns:
            A fully populated ``ItemVerification`` with deterministic
            ``confidence`` and ``notes``.

        Raises:
            agents.OutputGuardrailTripwireTriggered: When the Atwater, schema,
                or confidence guardrail detects an inconsistency.
            agents.MaxTurnsExceeded: When the agent exceeds the turn limit.
        """
        self._logger.info("verifier_adapter.start", item_id=item.id, name=item.name)

        context = VerifierContext(
            usda=self._usda,
            off=self._off,
            tavily=self._tavily,
            settings=self._settings,
        )

        intro = "Verify the following food item and return a complete ItemVerification:"
        prompt = f"{intro}\n\n{item.model_dump_json(indent=2)}"

        result = await Runner.run(
            self._agent,
            input=prompt,
            context=context,
        )

        verification: ItemVerification = result.final_output_as(
            ItemVerification, raise_if_incorrect_type=True,
        )

        # ------------------------------------------------------------------
        # Post-run overrides — applied regardless of guardrail outcome
        # ------------------------------------------------------------------
        overrides: dict[str, Any] = {}

        # Replace LLM-generated notes with mechanically-derived tool events.
        # This kills the fabrication channel: the LLM can no longer invent
        # error messages that did not occur.
        derived_notes = list(context.tool_events)
        if derived_notes != list(verification.notes):
            overrides["notes"] = derived_notes

        # Override confidence with the deterministic rule so it always
        # reflects the actual evidence even if the guardrail did not trip.
        derived_conf = derive_confidence(verification)
        if derived_conf != verification.confidence:
            overrides["confidence"] = derived_conf

        if overrides:
            verification = verification.model_copy(update=overrides)

        self._logger.info(
            "verifier_adapter.done",
            item_id=item.id,
            verdict=verification.verdict.value,
            confidence=verification.confidence.value,
            evidence_count=len(verification.evidence),
        )

        return verification
