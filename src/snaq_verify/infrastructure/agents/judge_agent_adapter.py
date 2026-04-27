"""JudgeAgentAdapter — JudgeAgentPort backed by the OpenAI Agents SDK."""

from __future__ import annotations

from agents import Runner

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.eval_models import GroundTruthEntry, JudgeVerdict
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.ports.judge_agent_port import JudgeAgentPort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.infrastructure.agents.judge_agent import build_judge_agent


class JudgeAgentAdapter(JudgeAgentPort):
    """Implements ``JudgeAgentPort`` using an OpenAI Agents SDK ``Agent``.

    The adapter builds one judge ``Agent`` at construction time and reuses it
    for every ``judge`` call.  The judge agent has no tools — it performs a
    single structured-output inference that returns a ``JudgeVerdict``.

    The agent receives both the verifier's ``ItemVerification`` and the
    hand-curated ``GroundTruthEntry`` in a single prompt, then scores the
    verification against the ground truth.
    """

    def __init__(
        self,
        settings: Settings,
        logger: LoggerPort,
    ) -> None:
        """Construct the adapter.

        Args:
            settings: Application settings (model pin).
            logger: Structured logger for judging events.
        """
        self._settings = settings
        self._logger = logger
        self._agent = build_judge_agent(settings)

    # ------------------------------------------------------------------
    # JudgeAgentPort
    # ------------------------------------------------------------------

    async def judge(
        self,
        verification: ItemVerification,
        ground_truth: GroundTruthEntry,
    ) -> JudgeVerdict:
        """Score a verification against a golden-set entry.

        The verification and ground truth are serialised to JSON and combined
        into a single prompt.  The judge agent returns a ``JudgeVerdict``
        containing a score in [0, 1], a boolean for whether the verifier
        reached the correct verdict, and a free-text justification.

        Args:
            verification: The verifier agent's output for this item.
            ground_truth: The golden-set entry for the same item id.

        Returns:
            A ``JudgeVerdict`` with score, correct_verdict, and reasoning.

        Raises:
            agents.MaxTurnsExceeded: When the agent exceeds the turn limit.
        """
        self._logger.info(
            "judge_adapter.start",
            item_id=verification.item_id,
            verdict=verification.verdict.value,
        )

        prompt = (
            "Score the following verification against the ground truth.\n\n"
            "## Verification (verifier output)\n"
            f"{verification.model_dump_json(indent=2)}\n\n"
            "## Ground Truth (authoritative)\n"
            f"{ground_truth.model_dump_json(indent=2)}"
        )

        result = await Runner.run(
            self._agent,
            input=prompt,
        )

        verdict: JudgeVerdict = result.final_output_as(
            JudgeVerdict, raise_if_incorrect_type=True,
        )

        self._logger.info(
            "judge_adapter.done",
            item_id=verification.item_id,
            score=verdict.score,
            correct=verdict.correct_verdict,
        )

        return verdict
