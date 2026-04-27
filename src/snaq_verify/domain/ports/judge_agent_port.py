"""Judge agent port — LLM-as-judge for the eval layer."""

from abc import ABC, abstractmethod

from snaq_verify.domain.models.eval_models import GroundTruthEntry, JudgeVerdict
from snaq_verify.domain.models.item_verification import ItemVerification


class JudgeAgentPort(ABC):
    """Abstract interface for the LLM-as-judge eval agent.

    The adapter wraps an OpenAI Agents SDK `Agent` whose `output_type` is
    `JudgeVerdict`. The judge consumes the verifier's output for one item
    plus the corresponding hand-curated golden-set entry, and returns a
    score plus a reasoning string. No tools — pure structured-output call.
    """

    @abstractmethod
    async def judge(
        self,
        verification: ItemVerification,
        ground_truth: GroundTruthEntry,
    ) -> JudgeVerdict:
        """Score a verification against a golden-set entry.

        Args:
            verification: The verifier agent's output for this item.
            ground_truth: The golden-set entry for the same item id.

        Returns:
            A score in [0, 1], a boolean for "did the verifier reach the
            right verdict?", and a free-text justification.
        """
        raise NotImplementedError
