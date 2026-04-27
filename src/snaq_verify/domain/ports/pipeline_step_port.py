"""Pipeline step port — one stage of the verification or eval pipeline."""

from abc import ABC, abstractmethod

from snaq_verify.domain.models.pipeline_state import PipelineState


class PipelineStep(ABC):
    """Abstract interface for one pipeline step.

    Steps consume and produce `PipelineState`. They should read only the
    slices they need and write their step-specific output. Steps must be
    idempotent within a single pipeline run; running a step twice on the
    same state should be a no-op or produce the same output.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable, unique step identifier for logging and traces."""
        raise NotImplementedError

    @abstractmethod
    async def run(self, state: PipelineState) -> PipelineState:
        """Run this step against `state`, returning the updated state.

        Args:
            state: The current pipeline state.

        Returns:
            The updated pipeline state. Implementations may mutate `state`
            in place and return it, or return a new instance — callers must
            use the returned value, not the input.
        """
        raise NotImplementedError
