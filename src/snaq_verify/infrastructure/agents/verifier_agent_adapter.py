"""VerifierAgentAdapter — VerifierAgentPort backed by the OpenAI Agents SDK."""

from __future__ import annotations

from agents import Runner

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.open_food_facts_client_port import OpenFoodFactsClientPort
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort
from snaq_verify.domain.ports.usda_client_port import USDAClientPort
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort
from snaq_verify.infrastructure.agents.verifier_agent import (
    VerifierContext,
    build_verifier_agent,
)


class VerifierAgentAdapter(VerifierAgentPort):
    """Implements ``VerifierAgentPort`` using an OpenAI Agents SDK ``Agent``.

    The adapter builds one ``Agent`` at construction time and reuses it for
    every ``verify`` call.  Each call creates a fresh ``VerifierContext``
    so there is no shared mutable state between concurrent verifications.

    Tool calls are deterministic (numeric computations via ``function_tool``
    decorated functions); only the source-lookup tools (USDA / OFF / Tavily)
    perform I/O, and those are injected via the context so tests can swap
    them out without patching global state.
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
        ``ItemVerification``.  Two output guardrails fire before the result
        is returned.

        Args:
            item: The food item to verify.

        Returns:
            A fully populated ``ItemVerification``.

        Raises:
            agents.OutputGuardrailTripwireTriggered: When the Atwater or
                schema guardrail detects an inconsistency in the agent's output.
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

        self._logger.info(
            "verifier_adapter.done",
            item_id=item.id,
            verdict=verification.verdict.value,
            confidence=verification.confidence.value,
            evidence_count=len(verification.evidence),
        )

        return verification
