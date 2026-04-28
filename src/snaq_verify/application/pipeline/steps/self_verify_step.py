"""SelfVerifyStep — re-run verifier for low-scoring items using judge feedback."""

from __future__ import annotations

import asyncio

from snaq_verify.domain.models.eval_models import GroundTruthEntry
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.judge_agent_port import JudgeAgentPort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort

# Items whose judge score falls below this threshold get a re-run.
_RETRY_SCORE_THRESHOLD = 0.5


class SelfVerifyStep(PipelineStep):
    """Re-verify low-quality items using judge feedback as a hint.

    For each item that already has a verification in ``state.verifications``,
    this step runs the judge inline.  If the score is below the threshold, it
    calls the verifier a second time — passing the judge's ``reasoning`` as a
    ``hint`` — so the agent can self-correct without repeating all tool calls.
    The improved verification replaces the original in ``state.verifications``.

    This step requires ``state.ground_truth`` to be populated (run
    ``LoadGroundTruthStep`` before ``VerifyStep`` in the pipeline).  Items
    without a matching ground-truth entry are silently skipped.

    Only intended for ``run-and-eval`` mode; the plain ``run`` command should
    omit this step to avoid the extra judge API calls.
    """

    def __init__(
        self,
        verifier_agent: VerifierAgentPort,
        judge_agent: JudgeAgentPort,
        logger: LoggerPort,
        retry_threshold: float = _RETRY_SCORE_THRESHOLD,
    ) -> None:
        """Construct the step.

        Args:
            verifier_agent: Port used to re-verify low-scoring items.
            judge_agent: Port used for inline judge scoring.
            logger: Structured logger for step events.
            retry_threshold: Judge score below which a retry is triggered.
                Defaults to 0.5.
        """
        self._verifier = verifier_agent
        self._judge = judge_agent
        self._logger = logger
        self._threshold = retry_threshold

    @property
    def name(self) -> str:
        """Step identifier."""
        return "self_verify"

    async def run(self, state: PipelineState) -> PipelineState:
        """Re-verify low-scoring items using inline judge feedback.

        Args:
            state: Pipeline state; ``verifications`` and ``ground_truth``
                should already be populated.

        Returns:
            Updated state with low-scoring verifications replaced by improved
            ones where possible.
        """
        if not state.verifications:
            self._logger.info("self_verify.skipped_no_verifications")
            return state

        gt_map: dict[str, GroundTruthEntry] = {
            e.item_id: e for e in state.ground_truth
        }
        item_map = {item.id: item for item in state.items}

        async def _maybe_retry(
            verification: ItemVerification,
        ) -> ItemVerification:
            gt = gt_map.get(verification.item_id)
            if gt is None:
                # No ground-truth entry for this item — skip silently.
                return verification

            # 1. Judge inline to get a quality signal.
            verdict = await self._judge.judge(verification, gt)
            self._logger.info(
                "self_verify.judge_inline",
                item_id=verification.item_id,
                score=verdict.score,
            )

            if verdict.score >= self._threshold:
                return verification  # Good enough — keep original.

            # 2. Score too low — retry with judge feedback as a hint.
            item = item_map[verification.item_id]
            hint = (
                f"Prior verification score: {verdict.score:.2f}/1.0. "
                f"Judge feedback: {verdict.reasoning} "
                f"Please re-examine your evidence and improve the verdict."
            )
            self._logger.info(
                "self_verify.retry",
                item_id=verification.item_id,
                score=verdict.score,
                hint_preview=hint[:80],
            )
            return await self._verifier.verify(item, hint=hint)

        # Run all judge calls + retries concurrently (already throttled by
        # the underlying adapter's semaphore / rate limits).
        improved_list: list[ItemVerification] = list(
            await asyncio.gather(
                *[_maybe_retry(v) for v in state.verifications],
            ),
        )

        retried = sum(
            1
            for old, new in zip(state.verifications, improved_list, strict=True)
            if old is not new
        )
        state.verifications = improved_list
        self._logger.info(
            "self_verify.done", retried=retried, total=len(improved_list),
        )
        return state
