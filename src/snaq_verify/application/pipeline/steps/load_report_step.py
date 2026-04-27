"""LoadReportStep — eval-only; reads an existing verification report from disk."""

import json

from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.models.verification_report import VerificationReport
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class LoadReportStep(PipelineStep):
    """Populate `state.report` and `state.verifications` for the eval pipeline.

    When running the combined `run-and-eval` command the verification report
    is already in memory; this step is a no-op in that case.  When running
    `eval` standalone it reads the JSON from `state.output_path` (the path
    passed via ``--report``).
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
        return "load_report"

    async def run(self, state: PipelineState) -> PipelineState:
        """Load the verification report into state.

        If `state.report` is already populated (run-and-eval scenario) this
        step is a no-op.  Otherwise reads from `state.output_path`.

        Args:
            state: Pipeline state.

        Returns:
            Updated state with `report` and `verifications` populated.

        Raises:
            ValueError: If `output_path` is None when report is not in memory.
            FileNotFoundError: If the report file does not exist.
            ValueError: If the JSON is invalid or fails schema validation.
        """
        if state.report is not None:
            self._logger.info("load_report.using_in_memory_report")
            state.verifications = list(state.report.items)
            return state

        if state.output_path is None:
            raise ValueError(
                "LoadReportStep requires state.output_path when report is not in memory"
            )

        path = state.output_path
        self._logger.info("load_report.reading", path=str(path))

        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._logger.error("load_report.file_not_found", path=str(path))
            raise

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        state.report = VerificationReport.model_validate(data)
        state.verifications = list(state.report.items)
        self._logger.info(
            "load_report.done",
            path=str(path),
            item_count=len(state.verifications),
        )
        return state
