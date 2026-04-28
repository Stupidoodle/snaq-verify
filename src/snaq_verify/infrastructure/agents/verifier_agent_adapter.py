"""VerifierAgentAdapter — VerifierAgentPort backed by the OpenAI Agents SDK."""

from __future__ import annotations

from typing import Any

from agents import ReasoningItem, Runner, TResponseInputItem

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.enums import ConfidenceLevel
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

_LOW_CONFIDENCE_RETRY_PROMPT = (
    "Your confidence is LOW. Re-examine the evidence you already gathered. "
    "If you have ≥2 sources with nutrition data, re-score the candidate "
    "match and update your verdict. Do not call any search tools again — "
    "use only the tool results already in this conversation."
)


class VerifierAgentAdapter(VerifierAgentPort):
    """Implements ``VerifierAgentPort`` using an OpenAI Agents SDK ``Agent``.

    The adapter builds one ``Agent`` at construction time and reuses it for
    every ``verify`` call.  Each call creates a fresh ``VerifierContext``
    so there is no shared mutable state between concurrent verifications.

    **Post-run overrides** applied after ``Runner.run`` returns:

    - *notes*: Replaced with ``context.tool_events`` — a mechanically-derived
      list of notable IO events (404s, empty results, web-search fallbacks).
      Prevents LLM from fabricating plausible-sounding error messages.

    - *confidence*: Re-derived deterministically from evidence (number of
      sources, top match score, overall verdict).  This is the authoritative
      value.

    - *reasoning*: Native reasoning tokens (``ReasoningItem.raw_item.summary``)
      override the LLM's self-reported field when present.

    **Retry logic**: When the first run produces ``ConfidenceLevel.LOW``, the
    adapter sends one follow-up message asking the agent to re-evaluate using
    the evidence already in the conversation (no new tool calls).  At most one
    retry to limit token cost.
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

    async def verify(self, item: FoodItem, hint: str | None = None) -> ItemVerification:
        """Verify a single food item using the verifier agent.

        The item is serialised to JSON and passed as the agent's input message.
        The agent calls USDA / OFF / Tavily tools to gather evidence, then
        runs deterministic compute tools to produce a structured
        ``ItemVerification``.  Three output guardrails fire before the result
        is returned.

        If the first run produces ``ConfidenceLevel.LOW``, the adapter retries
        once with a feedback prompt that asks the agent to re-evaluate using
        already-gathered evidence (no extra tool calls).  ``tool_events``
        accumulates across both runs so the audit trail remains complete.

        After all runs, the adapter applies three deterministic overrides:

        1. ``notes`` ← ``context.tool_events`` (mechanically-derived audit
           trail, prevents LLM fabrication).
        2. ``confidence`` ← ``derive_confidence(verification)`` (exact rule
           from instructions, defeats any LLM deviation).
        3. ``reasoning`` ← native ``ReasoningItem`` summaries when present
           (more authoritative than LLM self-report).

        Args:
            item: The food item to verify.
            hint: Optional feedback from a prior judge run prepended to the
                prompt.  Used by ``SelfVerifyStep`` to guide re-verification.
                Defaults to ``None`` (standard verification).

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
            item=item,
        )

        intro = "Verify the following food item and return a complete ItemVerification:"
        if hint:
            intro = f"HINT FROM PRIOR EVAL: {hint}\n\n{intro}"
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
        # Optional retry on LOW confidence
        # ------------------------------------------------------------------
        if derive_confidence(verification) is ConfidenceLevel.LOW:
            self._logger.info(
                "verifier_adapter.retry",
                item_id=item.id,
                reason="low_confidence",
            )
            input_items: list[TResponseInputItem] = result.to_input_list()
            input_items.append(
                {"role": "user", "content": _LOW_CONFIDENCE_RETRY_PROMPT},
            )
            retry_result = await Runner.run(
                self._agent,
                input=input_items,
                context=context,
            )
            verification = retry_result.final_output_as(
                ItemVerification, raise_if_incorrect_type=True,
            )
            # Repoint result to retry result for reasoning extraction below
            result = retry_result

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

        # Extract native reasoning tokens when the model produces them
        # (e.g. gpt-5.1 with reasoning=Reasoning(effort="low", summary="auto")).
        # When present, the SDK's structured reasoning summary is more reliable
        # than the LLM's self-reported `reasoning` field in the output schema.
        # If the model doesn't produce reasoning items, the LLM-populated field
        # from `output_type` is kept unchanged (or stays None if not set).
        native_reasoning: str | None = (
            "\n\n".join(
                s.text
                for run_item in result.new_items
                if isinstance(run_item, ReasoningItem)
                for s in run_item.raw_item.summary
            )
            or None
        )
        if native_reasoning is not None:
            overrides["reasoning"] = native_reasoning

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
