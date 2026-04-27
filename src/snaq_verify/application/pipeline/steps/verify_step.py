"""VerifyStep — calls the verifier agent for each item concurrently."""

import asyncio
from collections.abc import Callable

from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort

_DEFAULT_CONCURRENCY = 1


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
                verification = await self._verifier_agent.verify(item)
                result_slots[idx] = verification
                self._logger.info(
                    "verify.item_done",
                    item_id=item.id,
                    verdict=verification.verdict.value,
                )
                if self._on_item_complete is not None:
                    self._on_item_complete(item.id)

        await asyncio.gather(
            *[_verify_one(i, item) for i, item in enumerate(state.items)]
        )

        # Reconstruct in input order.
        state.verifications = [result_slots[i] for i in range(len(state.items))]
        self._logger.info("verify.done", total=len(state.verifications))
        return state
