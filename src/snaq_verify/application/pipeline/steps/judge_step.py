"""JudgeStep — eval-only; scores each verification against ground truth."""

from datetime import UTC, datetime

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.eval_models import (
    EvalReport,
    EvalRunMetadata,
    GroundTruthEntry,
    JudgeVerdict,
)
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.judge_agent_port import JudgeAgentPort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class JudgeStep(PipelineStep):
    """Run the LLM judge against every verification that has a ground-truth entry.

    Items without a matching `item_id` in `state.ground_truth` are silently
    skipped.  The step aggregates all `JudgeVerdict` objects into
    `state.eval_report`.
    """

    def __init__(
        self,
        judge_agent: JudgeAgentPort,
        logger: LoggerPort,
        settings: Settings,
    ) -> None:
        """Construct the step.

        Args:
            judge_agent: Port implementation for the LLM-as-judge.
            logger: Structured logger for step events.
            settings: Application settings (provides the model pin).
        """
        self._judge_agent = judge_agent
        self._logger = logger
        self._settings = settings

    @property
    def name(self) -> str:
        """Step identifier."""
        return "judge"

    async def run(self, state: PipelineState) -> PipelineState:
        """Judge all verifications that have matching ground-truth entries.

        Args:
            state: Pipeline state; `report` and `ground_truth` must be set.

        Returns:
            Updated state with `eval_report` populated.

        Raises:
            ValueError: If `state.report` is None.
        """
        if state.report is None:
            raise ValueError(
                "JudgeStep requires state.report — run LoadReportStep first"
            )

        gt_map: dict[str, GroundTruthEntry] = {
            entry.item_id: entry for entry in state.ground_truth
        }

        judgments: list[JudgeVerdict] = []
        for verification in state.report.items:
            gt_entry = gt_map.get(verification.item_id)
            if gt_entry is None:
                self._logger.warning(
                    "judge.no_ground_truth",
                    item_id=verification.item_id,
                )
                continue

            self._logger.info("judge.item_start", item_id=verification.item_id)
            verdict = await self._judge_agent.judge(verification, gt_entry)
            judgments.append(verdict)
            self._logger.info(
                "judge.item_done",
                item_id=verification.item_id,
                score=verdict.score,
                correct=verdict.correct_verdict,
            )

        total = len(judgments)
        aggregate_score = (sum(j.score for j in judgments) / total) if total > 0 else 0.0
        correct_verdicts = sum(1 for j in judgments if j.correct_verdict)

        state.eval_report = EvalReport(
            metadata=EvalRunMetadata(
                timestamp=datetime.now(UTC),
                model=self._settings.OPENAI_MODEL,
                item_count=total,
            ),
            judgments=judgments,
            aggregate_score=aggregate_score,
            correct_verdicts=correct_verdicts,
            total=total,
        )

        self._logger.info(
            "judge.done",
            total=total,
            aggregate_score=round(aggregate_score, 3),
            correct_verdicts=correct_verdicts,
        )
        return state
