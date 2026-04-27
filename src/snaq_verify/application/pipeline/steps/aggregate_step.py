"""AggregateStep — compose VerificationReport from items + verifications."""

from datetime import UTC, datetime

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.models.verification_report import RunMetadata, VerificationReport
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep

_FLAG_VERDICTS = frozenset({Verdict.MINOR_DISCREPANCY, Verdict.MAJOR_DISCREPANCY})


class AggregateStep(PipelineStep):
    """Compose a `VerificationReport` from `state.items` + `state.verifications`.

    Sets `RunMetadata` (timestamp, input_count, flag_count, model) and writes
    the assembled report to `state.report`.  The timestamp is the only field
    that varies between reruns on the same input — used by the determinism
    check.
    """

    def __init__(self, logger: LoggerPort, settings: Settings) -> None:
        """Construct the step.

        Args:
            logger: Structured logger for step events.
            settings: Application settings (provides the model pin).
        """
        self._logger = logger
        self._settings = settings

    @property
    def name(self) -> str:
        """Step identifier."""
        return "aggregate"

    async def run(self, state: PipelineState) -> PipelineState:
        """Build `state.report` from accumulated verifications.

        Args:
            state: Pipeline state; `items` and `verifications` must be set.

        Returns:
            Updated state with `report` populated.
        """
        flag_count = sum(
            1 for v in state.verifications if v.verdict in _FLAG_VERDICTS
        )

        metadata = RunMetadata(
            timestamp=datetime.now(UTC),
            input_count=len(state.items),
            flag_count=flag_count,
            model=self._settings.OPENAI_MODEL,
        )

        state.report = VerificationReport(
            metadata=metadata,
            items=list(state.verifications),
        )

        self._logger.info(
            "aggregate.done",
            input_count=metadata.input_count,
            flag_count=flag_count,
            model=metadata.model,
        )
        return state
