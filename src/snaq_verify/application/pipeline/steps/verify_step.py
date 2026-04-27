"""VerifyStep — calls the verifier agent for each item concurrently."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

from snaq_verify.application.tools.check_atwater_consistency import (
    check_atwater_consistency,
)
from snaq_verify.domain.models.atwater_check import AtwaterCheck
from snaq_verify.domain.models.enums import ConfidenceLevel, Verdict
from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort

_DEFAULT_CONCURRENCY = 1


def _fallback_verification(
    item: FoodItem, reason: str, exc: Exception,
) -> ItemVerification:
    """Build a degraded ItemVerification when the agent fails for one item.

    Used to keep the pipeline moving when a guardrail tripwire fires or the
    agent raises an unexpected error — better to flag the item as
    LOW_CONFIDENCE / NO_DATA and finish the batch than to abort the entire
    11-item run.
    """
    atwater = check_atwater_consistency(
        item.nutrition_per_100g, tolerance_pct=15.0,
    )
    return ItemVerification(
        item_id=item.id,
        item_name=item.name,
        reported_nutrition=item.nutrition_per_100g,
        verdict=Verdict.NO_DATA,
        confidence=ConfidenceLevel.LOW,
        evidence=[],
        proposed_correction=None,
        atwater_check_input=AtwaterCheck.model_validate(atwater.model_dump()),
        summary=(
            f"{item.name}: agent run failed ({reason}). "
            f"No verdict produced; pipeline continued."
        ),
        notes=[
            f"agent_failure_{reason}: {type(exc).__name__}: {str(exc)[:200]}",
            f"timestamp_utc: {datetime.now(UTC).isoformat()}",
        ],
    )


class VerifyStep(PipelineStep):
    """Verify every food item by calling the verifier agent.

    Items are dispatched concurrently but throttled by a semaphore so we
    never hammer the USDA API with more than `concurrency` simultaneous
    requests.  Output order mirrors the *input* order regardless of
    completion order — deterministic report layout is guaranteed.

    An optional `on_item_complete` callback is invoked after each item
    finishes; the CLI uses this to advance a Rich progress bar.
    """

    def __init__(
        self,
        verifier_agent: VerifierAgentPort,
        logger: LoggerPort,
        concurrency: int = _DEFAULT_CONCURRENCY,
        on_item_complete: Callable[[str], None] | None = None,
    ) -> None:
        """Construct the step.

        Args:
            verifier_agent: Port implementation used to verify each item.
            logger: Structured logger for step events.
            concurrency: Maximum simultaneous verification calls.
            on_item_complete: Optional callback called with the item_id once
                each verification finishes.  Used for CLI progress display.
        """
        self._verifier_agent = verifier_agent
        self._logger = logger
        self._concurrency = concurrency
        self._on_item_complete = on_item_complete

    @property
    def name(self) -> str:
        """Step identifier."""
        return "verify"

    async def run(self, state: PipelineState) -> PipelineState:
        """Verify all items in `state.items`, storing results in-order.

        Args:
            state: Pipeline state; `items` should be populated by
                `LoadInputStep`.

        Returns:
            Updated state with `verifications` populated in input order.
        """
        if not state.items:
            self._logger.info("verify.skipped_empty_input")
            state.verifications = []
            return state

        semaphore = asyncio.Semaphore(self._concurrency)
        # Pre-allocate a slot per item so results land in input order.
        result_slots: dict[int, ItemVerification] = {}

        async def _verify_one(idx: int, item: FoodItem) -> None:
            async with semaphore:
                self._logger.info("verify.item_start", item_id=item.id, index=idx)
                try:
                    verification = await self._verifier_agent.verify(item)
                except (
                    OutputGuardrailTripwireTriggered,
                    InputGuardrailTripwireTriggered,
                ) as exc:
                    self._logger.warning(
                        "verify.item_guardrail_trip",
                        item_id=item.id,
                        error=str(exc)[:200],
                    )
                    verification = _fallback_verification(item, "guardrail_trip", exc)
                except Exception as exc:  # noqa: BLE001 — pipeline must not abort
                    self._logger.error(
                        "verify.item_error",
                        item_id=item.id,
                        error_type=type(exc).__name__,
                        error=str(exc)[:200],
                    )
                    verification = _fallback_verification(item, "agent_error", exc)
                result_slots[idx] = verification
                self._logger.info(
                    "verify.item_done",
                    item_id=item.id,
                    verdict=verification.verdict.value,
                )
                if self._on_item_complete is not None:
                    self._on_item_complete(item.id)

        await asyncio.gather(
            *[_verify_one(i, item) for i, item in enumerate(state.items)],
        )

        # Reconstruct in input order.
        state.verifications = [result_slots[i] for i in range(len(state.items))]
        self._logger.info("verify.done", total=len(state.verifications))
        return state
