"""WriteReportStep — serialize VerificationReport to JSON on disk."""

from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class WriteReportStep(PipelineStep):
    """Write `state.report` as indented JSON to `state.output_path`.

    Creates parent directories if they do not exist.  Overwrites any
    existing file at `output_path` — callers are responsible for
    choosing a safe destination.
    """

    def __init__(self, logger: LoggerPort) -> None:
        """Construct the step.

        Args:
            logger: Structured logger for step events.
        """
        self._logger = logger

    @property
    def name(self) -> str:
        """Step identifier."""
        return "write_report"

    async def run(self, state: PipelineState) -> PipelineState:
        """Serialize and write the verification report.

        Args:
            state: Pipeline state; both `report` and `output_path` must be
                set.

        Returns:
            Unchanged state (side-effect: file written to disk).

        Raises:
            ValueError: If `report` or `output_path` is None.
            OSError: If the file cannot be written.
        """
        if state.report is None:
            raise ValueError(
                "WriteReportStep requires state.report — run AggregateStep first"
            )
        if state.output_path is None:
            raise ValueError("WriteReportStep requires state.output_path to be set")

        path = state.output_path
        self._logger.info("write_report.writing", path=str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.report.model_dump_json(indent=2), encoding="utf-8")

        self._logger.info(
            "write_report.done",
            path=str(path),
            bytes=path.stat().st_size,
        )
        return state
