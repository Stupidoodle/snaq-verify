"""LoadGroundTruthStep — eval-only; reads ground_truth.json into state."""

import json

from snaq_verify.domain.models.eval_models import GroundTruthEntry
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class LoadGroundTruthStep(PipelineStep):
    """Read `state.ground_truth_path` and populate `state.ground_truth`.

    The file must be a JSON array whose objects conform to `GroundTruthEntry`.
    Used by the eval pipeline before `JudgeStep`.
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
        return "load_ground_truth"

    async def run(self, state: PipelineState) -> PipelineState:
        """Read the ground-truth file into `state.ground_truth`.

        Args:
            state: Pipeline state; `ground_truth_path` must be set.

        Returns:
            Updated state with `ground_truth` populated.

        Raises:
            ValueError: If `ground_truth_path` is None.
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or fails schema validation.
        """
        if state.ground_truth_path is None:
            raise ValueError(
                "LoadGroundTruthStep requires state.ground_truth_path to be set"
            )

        path = state.ground_truth_path
        self._logger.info("load_ground_truth.reading", path=str(path))

        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._logger.error("load_ground_truth.file_not_found", path=str(path))
            raise

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in {path}, got {type(data).__name__}"
            )

        state.ground_truth = [GroundTruthEntry.model_validate(entry) for entry in data]
        self._logger.info(
            "load_ground_truth.done",
            count=len(state.ground_truth),
            path=str(path),
        )
        return state
