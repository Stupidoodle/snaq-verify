"""FakeJudgeAgent — in-memory JudgeAgentPort for unit tests."""

from snaq_verify.domain.models.eval_models import GroundTruthEntry, JudgeVerdict
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.ports.judge_agent_port import JudgeAgentPort


class FakeJudgeAgent(JudgeAgentPort):
    """Returns configurable JudgeVerdict results without any LLM calls.

    By default, each call returns a perfect score (1.0, correct_verdict=True)
    for the given item.  Override via `fixed_result` or `results_by_id`.
    """

    def __init__(
        self,
        fixed_result: JudgeVerdict | None = None,
        results_by_id: dict[str, JudgeVerdict] | None = None,
    ) -> None:
        self._fixed_result = fixed_result
        self._results_by_id = results_by_id or {}
        self.calls: list[tuple[ItemVerification, GroundTruthEntry]] = []

    async def judge(
        self,
        verification: ItemVerification,
        ground_truth: GroundTruthEntry,
    ) -> JudgeVerdict:
        """Return a pre-configured or default JudgeVerdict."""
        self.calls.append((verification, ground_truth))
        if verification.item_id in self._results_by_id:
            return self._results_by_id[verification.item_id]
        if self._fixed_result is not None:
            return self._fixed_result
        return JudgeVerdict(
            item_id=verification.item_id,
            score=1.0,
            correct_verdict=True,
            reasoning="FakeJudgeAgent: default perfect score.",
        )
