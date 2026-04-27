"""WriteEvalReportStep — serialize EvalReport to JSON on disk."""

from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class WriteEvalReportStep(PipelineStep):
    """Write `state.eval_report` as indented JSON to `state.eval_output_path`.

    Creates parent directories if they do not exist.  Overwrites any
    existing file at `eval_output_path`.
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
        return "write_eval_report"

    async def run(self, state: PipelineState) -> PipelineState:
        """Serialize and write the eval report.

        Args:
            state: Pipeline state; both `eval_report` and `eval_output_path`
                must be set.

        Returns:
            Unchanged state (side-effect: file written to disk).

        Raises:
            ValueError: If `eval_report` or `eval_output_path` is None.
            OSError: If the file cannot be written.
        """
        if state.eval_report is None:
            raise ValueError(
                "WriteEvalReportStep requires state.eval_report — run JudgeStep first"
            )
        if state.eval_output_path is None:
            raise ValueError(
                "WriteEvalReportStep requires state.eval_output_path to be set"
            )

        path = state.eval_output_path
        self._logger.info("write_eval_report.writing", path=str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.eval_report.model_dump_json(indent=2), encoding="utf-8")

        self._logger.info(
            "write_eval_report.done",
            path=str(path),
            bytes=path.stat().st_size,
        )
        return state
