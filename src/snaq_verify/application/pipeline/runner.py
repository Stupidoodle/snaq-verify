"""Pipeline runner — execute an ordered list of `PipelineStep`s."""

from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class PipelineRunner:
    """Sequentially execute pipeline steps against a single state object.

    Each step's start and completion is logged with structured context.
    Exceptions propagate to the caller — the CLI decides how to surface
    them. Determinism is the responsibility of each step; the runner only
    enforces order.
    """

    def __init__(self, logger: LoggerPort) -> None:
        """Construct a runner.

        Args:
            logger: Structured logger used for step-boundary events.
        """
        self._logger = logger

    async def run(
        self,
        state: PipelineState,
        steps: list[PipelineStep],
    ) -> PipelineState:
        """Run `steps` in order against `state`.

        Args:
            state: Initial pipeline state.
            steps: Ordered list of steps to execute.

        Returns:
            The state after every step has run.
        """
        for step in steps:
            self._logger.info("pipeline_step.start", step=step.name)
            state = await step.run(state)
            self._logger.info("pipeline_step.complete", step=step.name)
        return state
